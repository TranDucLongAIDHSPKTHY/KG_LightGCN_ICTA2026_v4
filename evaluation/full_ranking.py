"""
evaluation/full_ranking.py
─────────────────────────────────────────────────────────────────────────────
Full-ranking evaluation — tối ưu cho dataset lớn (50K users, 90K items).

Chiến lược:
  1. Tính toàn bộ user/item embeddings một lần (get_embeddings)
  2. Score theo batch users — tránh OOM với dataset lớn
  3. Dùng torch.topk(k=max_k) thay vì full sort — nhanh hơn nhiều
  4. Train mask dùng scatter_ trên tensor thay vì Python loop
  5. Chỉ score users có ground-truth trong eval split (bỏ qua user rỗng)
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from evaluation.metrics import compute_all_metrics


@torch.no_grad()
def full_ranking_eval(
    model,
    train_user2items: Dict[int, List[int]],
    eval_user2items:  Dict[int, List[int]],
    n_items: int,
    device:  torch.device,
    batch_size: int = 2048,
    top_k_list: List[int] = [10, 20],
) -> Dict[str, float]:
    """
    Full-item ranking evaluation (tối ưu cho dataset lớn).

    Args:
        model:              Model có get_embeddings() → (user_emb, item_emb).
        train_user2items:   user → train items (để mask khi rank).
        eval_user2items:    user → eval items  (ground truth).
        n_items:            Tổng số items.
        device:             Thiết bị tính toán.
        batch_size:         Số user xử lý một lúc (tăng lên nếu GPU đủ VRAM).
        top_k_list:         Danh sách K (vd [10, 20]).

    Returns:
        {metric_name: float}
    """
    model.eval()
    max_k = max(top_k_list)

    # ── Lấy embedding một lần ─────────────────────────────────────────────────
    # get_embeddings() should already be wrapped with @torch.no_grad in models.
    user_emb, item_emb = model.get_embeddings()
    item_emb = item_emb.to(device)   # [M, D] — keep on GPU for matmul
    user_emb = user_emb.to(device)   # [N, D] — indexed by batch

    # ── Chỉ xử lý users có ground-truth (tránh lãng phí compute) ─────────────
    eval_users = [u for u in eval_user2items if eval_user2items[u]]
    if not eval_users:
        return {}
    eval_users_arr = np.array(eval_users, dtype=np.int64)

    all_ranked: List[np.ndarray] = []
    all_gt:     List[List[int]]  = []

    for start in range(0, len(eval_users_arr), batch_size):
        batch_uids = eval_users_arr[start: start + batch_size]   # [B]
        B = len(batch_uids)

        u_emb = user_emb[batch_uids]                             # [B, D]

        # Score [B, M]
        scores = torch.matmul(u_emb, item_emb.T)                # [B, M]

        # ── Mask train items bằng scatter (không dùng Python loop) ───────────
        # Xây dựng (flat_user_idx, flat_item_idx) cho tất cả train items
        flat_local = []   # index trong batch (0..B-1)
        flat_items = []   # item ids

        for local_i, uid in enumerate(batch_uids.tolist()):
            items = train_user2items.get(uid, [])
            if items:
                flat_local.extend([local_i] * len(items))
                flat_items.extend(items)

        if flat_local:
            li = torch.tensor(flat_local, dtype=torch.long, device=device)
            fi = torch.tensor(flat_items, dtype=torch.long, device=device)
            scores[li, fi] = float("-inf")

        # ── Top-K ──────────────────────────────────────────────────────────────
        k_eff = min(max_k, n_items)
        _, ranked = torch.topk(scores, k=k_eff, dim=-1, largest=True, sorted=True)
        ranked_np = ranked.cpu().numpy()    # [B, k_eff]

        for local_i, uid in enumerate(batch_uids.tolist()):
            all_ranked.append(ranked_np[local_i])
            all_gt.append(eval_user2items[uid])

    ranked_matrix = np.vstack(all_ranked)   # [N_eval, k_eff]

    # Free memory after evaluation (works for both CPU and GPU)
    del user_emb, item_emb
    if device.type == "cuda":
        torch.cuda.empty_cache()
    import gc; gc.collect()

    return compute_all_metrics(ranked_matrix, all_gt, top_k_list)

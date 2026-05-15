"""
losses/bpr_loss.py
─────────────────────────────────────────────────────────────────────────────
Bayesian Personalised Ranking (BPR) loss.
Rendle et al., UAI 2009 — https://arxiv.org/abs/1205.2618

L_BPR = -Σ log σ(ŷ_ui+ - ŷ_ui-)
"""

import torch
import torch.nn.functional as F


def bpr_loss(
    user_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Compute BPR loss given user, positive, and negative embeddings.

    Args:
        user_emb: [B, D] user embeddings.
        pos_emb:  [B, D] positive item embeddings.
        neg_emb:  [B, D] negative item embeddings.

    Returns:
        Scalar BPR loss (mean over batch).
    """
    pos_scores = (user_emb * pos_emb).sum(dim=-1)  # [B]
    neg_scores = (user_emb * neg_emb).sum(dim=-1)  # [B]
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    return loss


def kg_bpr_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> torch.Tensor:
    """
    BPR loss for KG embedding (TransR).

    Args:
        pos_scores: [B] positive triple scores (lower = better in TransR).
        neg_scores: [B] negative triple scores.

    Returns:
        Scalar BPR loss.
    """
    # TransR: minimise ||h+r-t_pos|| and maximise ||h+r-t_neg||
    loss = -F.logsigmoid(neg_scores - pos_scores).mean()
    return loss

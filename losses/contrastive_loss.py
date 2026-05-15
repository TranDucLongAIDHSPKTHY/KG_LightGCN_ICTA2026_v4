"""
losses/contrastive_loss.py
─────────────────────────────────────────────────────────────────────────────
InfoNCE / NT-Xent contrastive loss used by SimGCL and KGCL.

L_CL = -log [ exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) ]
"""

import torch
import torch.nn.functional as F


def infonce_loss(
    view1: torch.Tensor,
    view2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    Bidirectional InfoNCE (NT-Xent) contrastive loss.

    Args:
        view1:       [B, D] first augmented view.
        view2:       [B, D] second augmented view.
        temperature: Softmax temperature τ (fairness: 0.2).

    Returns:
        Scalar contrastive loss.
    """
    v1 = F.normalize(view1, dim=-1)   # [B, D]
    v2 = F.normalize(view2, dim=-1)   # [B, D]

    # Similarity matrix [B, B]
    sim = torch.matmul(v1, v2.T) / temperature

    # Positive pairs are on the diagonal
    labels = torch.arange(len(v1), device=v1.device)

    # Bidirectional: loss in both directions then average
    loss_12 = F.cross_entropy(sim, labels)
    loss_21 = F.cross_entropy(sim.T, labels)
    return (loss_12 + loss_21) / 2


def ssl_loss(
    emb_aug1: torch.Tensor,
    emb_aug2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    Alias of infonce_loss for clarity in trainer code.
    """
    return infonce_loss(emb_aug1, emb_aug2, temperature)

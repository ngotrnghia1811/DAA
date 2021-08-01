"""
All loss functions for the DAA framework.

L_total = L_task  +  λ_d·L_D  +  λ_w·L_W  +  λ_s·L_S  +  λ_m·L_M

  L_task  — Cross-entropy for event classification (§3.1)
  L_D     — Layer-wise domain-adversarial loss (§3.2, LDA)
  L_W     — Wasserstein distance estimation loss (§3.3)
  L_S     — Orthogonality / similarity loss (§3.2, ADD)
  L_M     — Masked Language Model auxiliary loss (§3.2, ADD)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Task loss
# ---------------------------------------------------------------------------

def event_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy event detection loss.

    Args:
        logits: (B, num_classes).
        labels: (B,) — ground-truth class IDs.

    Returns:
        Scalar cross-entropy loss.
    """
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Adapter-wise Domain Disentanglement (ADD) losses
# ---------------------------------------------------------------------------

def orthogonality_loss(
    joint_src: torch.Tensor,
    private_src: torch.Tensor,
    joint_tgt: torch.Tensor,
    private_tgt: torch.Tensor,
) -> torch.Tensor:
    """Frobenius-norm orthogonality constraint between joint and private adapters.

    L_S = ||A^j_s ⊤ A^s_s||_F²  +  ||A^j_t ⊤ A^t_t||_F²

    Args:
        joint_src:   (n^s, d) — joint adapter outputs on source data.
        private_src: (n^s, d) — source adapter outputs on source data.
        joint_tgt:   (n^t, d) — joint adapter outputs on target data.
        private_tgt: (n^t, d) — target adapter outputs on target data.

    Returns:
        Scalar similarity loss.
    """
    loss_src = torch.norm(joint_src.t() @ private_src, p="fro") ** 2
    loss_tgt = torch.norm(joint_tgt.t() @ private_tgt, p="fro") ** 2
    return loss_src + loss_tgt


def mlm_loss(
    mlm_logits: torch.Tensor,
    mlm_labels: torch.Tensor,
) -> torch.Tensor:
    """Masked Language Model cross-entropy loss.

    Args:
        mlm_logits: (B, T, V) — logits over vocabulary.
        mlm_labels: (B, T)    — target IDs (-100 for non-masked positions).

    Returns:
        Scalar MLM loss.
    """
    B, T, V = mlm_logits.shape
    return F.cross_entropy(
        mlm_logits.view(B * T, V),
        mlm_labels.view(B * T),
        ignore_index=-100,
    )

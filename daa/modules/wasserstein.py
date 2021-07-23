"""
Wasserstein distance estimator with gradient penalty (WGAN-GP style).

Estimates the Wasserstein-1 distance between source and target marginal
representation distributions using the domain-specific adapters A^s and A^t.

Estimation objective (maximised w.r.t. h_W):
    L̂_W = (1/n^s) Σ h_W(A^s(x^s_i))  −  (1/n^t) Σ h_W(A^t(x^t_i))

Gradient penalty (enforces Lipschitz constraint):
    L_GP(A^d) = (||∇_{A^d} h_W(A^d)||_2 − 1)²

Overall Wasserstein loss maximised w.r.t. h_W:
    L_W = L̂_W − λ_GP · L_GP

After estimation, source samples with the **lowest** h_W scores are
selected for LDA training of the joint adapter (they are most
transferable / closest to target distribution).

Reference:
    Shen et al. 2018 — Wasserstein Distance Guided Representation Learning
    Gulrajani et al. 2017 — Improved Training of Wasserstein GANs
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.autograd as autograd


class WassersteinEstimator(nn.Module):
    """Scalar critic network h_W : R^H → R.

    Input is the full-dimensional output (not the bottleneck) of the
    domain-specific adapter so that the distance operates on the same
    representation space as the LDA component.

    Args:
        input_dim:  Adapter output dimension (= BERT hidden size H).
        hidden_dim: Hidden layer size.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H) — representation from A^s or A^t.

        Returns:
            scores: (B, 1).
        """
        return self.net(x)


def wasserstein_loss(
    estimator: WassersteinEstimator,
    source_repr: torch.Tensor,
    target_repr: torch.Tensor,
) -> torch.Tensor:
    """Empirical Wasserstein distance (to be maximised w.r.t. h_W).

    Args:
        estimator:   h_W — the critic network.
        source_repr: (n^s, H) — A^s outputs for source samples.
        target_repr: (n^t, H) — A^t outputs for target samples.

    Returns:
        Scalar Wasserstein estimate (larger → more dissimilar).
    """
    src_scores = estimator(source_repr).mean()   # E[h_W(A^s(x^s))]
    tgt_scores = estimator(target_repr).mean()   # E[h_W(A^t(x^t))]
    return src_scores - tgt_scores


def gradient_penalty(
    estimator: WassersteinEstimator,
    source_repr: torch.Tensor,
    target_repr: torch.Tensor,
) -> torch.Tensor:
    """WGAN-GP gradient penalty.

    Interpolates between source and target representations and penalises
    the squared deviation of the gradient norm from 1.

    Args:
        estimator:   h_W.
        source_repr: (n, H).
        target_repr: (n, H).

    Returns:
        Scalar gradient penalty.
    """
    n = min(source_repr.size(0), target_repr.size(0))
    src = source_repr[:n]
    tgt = target_repr[:n]

    alpha = torch.rand(n, 1, device=src.device)
    interpolated = alpha * src + (1.0 - alpha) * tgt
    interpolated = interpolated.detach().requires_grad_(True)

    scores = estimator(interpolated)
    gradients = autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]  # (n, H)

    grad_norm = gradients.norm(2, dim=1)  # (n,)
    penalty = ((grad_norm - 1.0) ** 2).mean()
    return penalty


def select_source_samples(
    estimator: WassersteinEstimator,
    source_repr: torch.Tensor,
    selection_ratio: float = 0.7,
) -> torch.Tensor:
    """Return indices of the most-transferable source samples.

    The *lowest* h_W scores correspond to source samples geometrically
    closest to the target distribution.

    Args:
        estimator:       Trained h_W.
        source_repr:     (n^s, H) — A^s outputs for source batch.
        selection_ratio: Fraction of source samples to keep.

    Returns:
        selected_indices: 1-D LongTensor of selected sample indices.
    """
    with torch.no_grad():
        scores = estimator(source_repr).squeeze(-1)  # (n^s,)
    k = max(1, int(scores.size(0) * selection_ratio))
    _, indices = torch.topk(scores, k, largest=False, sorted=False)
    return indices

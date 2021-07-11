"""
Layer-wise Domain-Adversarial (LDA) component.

For every BERT layer l, a small binary domain classifier h^J_{d,l}
operates on the **down-projected** (bottleneck) representation of the
joint adapter A^j.  The asymmetric relaxation of DANN is used:

    L_D^l = - (1/N) Σ [d_i · log(σ(h(a_l^{j,dw}(x_i))) / (1+β_l))
                       + (1-d_i) · log(1 - σ(h(a_l^{j,dw}(x_i))) / (1+β_l))]

with β_l = 2^(3-l)  (higher layers → smaller β → tighter alignment).

The domain classifier heads are updated in Step 1 of alternating
minimization; the joint adapter is updated in Step 2 (with reversed
domain labels to maximise divergence / fool the classifiers).

References:
    Wu et al. 2019 — Asymmetric tri-training for unsupervised domain adaptation
    Berto et al. 2021 — Linguistic Knowledge Meets Adaptation
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerDomainClassifier(nn.Module):
    """Binary domain classifier for one BERT layer's bottleneck representation.

    h^J_{d,l} : R^c → R  (scalar logit)
    """

    def __init__(self, bottleneck_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bottleneck: (B, T, c) — down-projected joint adapter output.

        Returns:
            logits: (B, T, 1).
        """
        return self.net(bottleneck)


class LayerWiseDomainAdversarial(nn.Module):
    """L domain classifiers (one per BERT layer) for LDA training.

    Args:
        num_layers:      Number of BERT transformer layers L.
        bottleneck_dim:  Dimension c of adapter down-projection.
        classifier_hidden: Hidden size inside each domain classifier.
    """

    def __init__(
        self,
        num_layers: int,
        bottleneck_dim: int,
        classifier_hidden: int = 128,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.classifiers = nn.ModuleList(
            [
                LayerDomainClassifier(bottleneck_dim, classifier_hidden)
                for _ in range(num_layers)
            ]
        )
        # β_l = 2^(3 - l), l is 1-indexed
        # layer_idx is 0-indexed so β[l] = 2^(3-(l+1)) = 2^(2-l)
        self.register_buffer(
            "betas",
            torch.tensor(
                [2.0 ** (3.0 - (l + 1)) for l in range(num_layers)], dtype=torch.float
            ),
        )

    def compute_lda_loss(
        self,
        bottlenecks: List[torch.Tensor],
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the sum of per-layer relaxed domain classification losses.

        To be MAXIMISED w.r.t. the classifier parameters and MINIMISED
        (via reversed domain labels) w.r.t. the joint adapter parameters.

        Args:
            bottlenecks:   List of L tensors, each (B, T, c),
                           where T is the trigger token dimension (after
                           index-selecting the trigger position).
            domain_labels: (B,) — 0 for source, 1 for target.

        Returns:
            Scalar total LDA loss (sum over layers).
        """
        total_loss = torch.tensor(0.0, device=domain_labels.device)
        N = domain_labels.size(0)
        d = domain_labels.float()  # (B,)

        for l, (classifier, bn_l) in enumerate(
            zip(self.classifiers, bottlenecks)
        ):
            beta_l = self.betas[l]
            # bn_l: (B, T, c)  — use trigger-token representation (T dimension squeezed)
            # If bn_l is (B, c), no squeeze needed; if (B, T, c) take trigger token.
            if bn_l.dim() == 3:
                bn_l = bn_l[:, 0, :]  # use [CLS] token as sentence representation

            logits = classifier(bn_l).squeeze(-1)  # (B,)
            probs = torch.sigmoid(logits) / (1.0 + beta_l)

            # Clamp for numerical stability
            eps = 1e-7
            probs = probs.clamp(eps, 1.0 - eps)

            loss_l = -(
                d * torch.log(probs) + (1.0 - d) * torch.log(1.0 - probs)
            ).mean()
            total_loss = total_loss + loss_l

        return total_loss

    def forward(
        self,
        bottlenecks: List[torch.Tensor],
        domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for compute_lda_loss (used in classifier update step)."""
        return self.compute_lda_loss(bottlenecks, domain_labels)

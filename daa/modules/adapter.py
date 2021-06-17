"""
Bottleneck adapter module for DAA.

Architecture (per BERT layer l):
    adapter_down : Linear(d_model → c)   +  GELU
    adapter_up   : Linear(c → d_model)
    output       : adapter_up(adapter_down(h)) + h  (skip connection)

where c << d_model  (controlled by ``bottleneck_dim``).

The DAA framework injects **three** independent adapters at every layer:
  - source adapter  A^s  (sees only source-domain inputs)
  - target adapter  A^t  (sees only target-domain inputs)
  - joint  adapter  A^j  (sees both; used for classification)

This module defines a single adapter stack (one per domain).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckAdapter(nn.Module):
    """One adapter module for a single BERT layer.

    Down-project → GELU → Up-project → residual.
    Also exposes the *down-projected* representation for layer-wise
    domain-adversarial training (as described in §3.2 of the paper).
    """

    def __init__(self, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        # Near-identity initialisation for the up-projection
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (batch, seq_len, hidden_size)  — FFN output of one BERT layer.

        Returns:
            (batch, seq_len, hidden_size)  — adapted hidden states.
        """
        down = F.gelu(self.down(hidden))   # (B, T, c)
        up = self.up(down)                 # (B, T, H)
        return up + hidden                 # skip connection

    def forward_with_bottleneck(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (adapted_output, bottleneck_representation).

        The bottleneck representation is used by domain adversarial heads
        for dimensionality-reduced alignment (§3.2 of the paper).
        """
        down = F.gelu(self.down(hidden))   # (B, T, c)
        up = self.up(down)                 # (B, T, H)
        return up + hidden, down


class DomainAdapterStack(nn.Module):
    """Stack of bottleneck adapters — one adapter per BERT layer for ONE domain.

    Usage:
        For each BERT layer l, feed the FFN output h_l through
        adapter_stack.adapters[l](h_l) to get the adapted representation.
    """

    def __init__(self, num_layers: int, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.adapters = nn.ModuleList(
            [BottleneckAdapter(hidden_size, bottleneck_dim) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward_layer(
        self, layer_idx: int, hidden: torch.Tensor
    ) -> torch.Tensor:
        """Apply the adapter at *layer_idx* to *hidden*."""
        return self.adapters[layer_idx](hidden)

    def forward_layer_with_bottleneck(
        self, layer_idx: int, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.adapters[layer_idx].forward_with_bottleneck(hidden)

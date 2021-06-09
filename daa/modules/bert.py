"""
Frozen BERT encoder that exposes per-layer hidden states.

The pre-trained BERT model is kept entirely frozen; only the adapter
modules (defined in adapter.py) are updated during training.

This module wraps HuggingFace's BertModel and provides a clean interface
that returns all intermediate hidden states needed for:
  - adapter input at each layer
  - domain-adversarial alignment
  - MLM-based self-supervised learning
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class FrozenBERTEncoder(nn.Module):
    """A BERT encoder whose parameters are permanently frozen.

    Attributes:
        bert: The underlying HuggingFace BertModel.
        config: BertConfig with model hyper-parameters.
        num_layers: Number of BERT transformer layers.
        hidden_size: Dimensionality of hidden states.
    """

    def __init__(self, bert_model_name: str, cache_dir: Optional[str] = None):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            bert_model_name,
            cache_dir=cache_dir,
            output_hidden_states=True,
        )
        # Freeze all BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.config: BertConfig = self.bert.config
        self.num_layers: int = self.config.num_hidden_layers
        self.hidden_size: int = self.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Run BERT and return a list of per-layer hidden states.

        Args:
            input_ids:      (B, T)  — token IDs.
            attention_mask: (B, T)  — 1 for real tokens, 0 for padding.

        Returns:
            hidden_states: list of (B, T, H) tensors, one per BERT layer
                           (index 0 = embedding layer, index 1..L = transformer layers).
                           Length is ``num_layers + 1``.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # hidden_states is a tuple: (embedding_output, layer_1, ..., layer_L)
        return list(outputs.hidden_states)

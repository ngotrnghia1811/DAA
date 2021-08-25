"""
Baseline models for comparison with DAA.

Three baselines from the paper:
  1. BERTBaseline  — fully fine-tuned BERT, no domain adaptation
  2. BERTPlusDAN   — BERT + DANN (gradient reversal layer)
  3. BERTPlusAdapter — BERT + single bottleneck adapter (no adaptation)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig
from daa.modules.adapter import DomainAdapterStack
from daa.modules.losses import event_classification_loss


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (for DANN)
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    """Identity in forward, scaled negation in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        alpha = ctx.saved_tensors[0].item()
        return -alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, alpha)


# ---------------------------------------------------------------------------
# Baseline 1: BERT (fully fine-tuned)
# ---------------------------------------------------------------------------

class BERTBaseline(nn.Module):
    """Cross-domain BERT fine-tuned only on source data (no adaptation)."""

    def __init__(
        self,
        bert_model_name: str,
        num_event_types: int,
        dropout: float = 0.1,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, cache_dir=cache_dir)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_event_types)

    def _trigger_repr(
        self, last_hidden: torch.Tensor, trigger_idx: torch.Tensor
    ) -> torch.Tensor:
        B, T, H = last_hidden.shape
        idx = trigger_idx.clamp(0, T - 1).unsqueeze(1).unsqueeze(2).expand(B, 1, H)
        return last_hidden.gather(1, idx).squeeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trigger_idx: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(self._trigger_repr(out.last_hidden_state, trigger_idx))
        logits = self.classifier(h)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = event_classification_loss(logits, labels)
        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trigger_idx: torch.Tensor,
    ) -> torch.Tensor:
        out = self.forward(input_ids, attention_mask, trigger_idx)
        return out["logits"].argmax(dim=-1)


# ---------------------------------------------------------------------------
# Baseline 2: BERT + DANN (gradient reversal)
# ---------------------------------------------------------------------------

class BERTPlusDANN(nn.Module):
    """BERT with DANN domain-adversarial training via gradient reversal."""

    def __init__(
        self,
        bert_model_name: str,
        num_event_types: int,
        dropout: float = 0.1,
        domain_hidden: int = 256,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, cache_dir=cache_dir)
        H = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.event_classifier = nn.Linear(H, num_event_types)
        self.domain_classifier = nn.Sequential(
            nn.Linear(H, domain_hidden),
            nn.ReLU(),
            nn.Linear(domain_hidden, 1),
        )

    def _trigger_repr(
        self, last_hidden: torch.Tensor, trigger_idx: torch.Tensor
    ) -> torch.Tensor:
        B, T, H = last_hidden.shape
        idx = trigger_idx.clamp(0, T - 1).unsqueeze(1).unsqueeze(2).expand(B, 1, H)
        return last_hidden.gather(1, idx).squeeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trigger_idx: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ) -> dict:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = self._trigger_repr(out.last_hidden_state, trigger_idx)
        h_drop = self.dropout(h)

        logits = self.event_classifier(h_drop)
        domain_logits = self.domain_classifier(grad_reverse(h_drop, alpha)).squeeze(-1)

        result = {"logits": logits, "domain_logits": domain_logits}

        if labels is not None:
            result["task_loss"] = event_classification_loss(logits, labels)
        if domain_labels is not None:
            result["domain_loss"] = F.binary_cross_entropy_with_logits(
                domain_logits, domain_labels.float()
            )
        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trigger_idx: torch.Tensor,
    ) -> torch.Tensor:
        out = self.forward(input_ids, attention_mask, trigger_idx)
        return out["logits"].argmax(dim=-1)


# ---------------------------------------------------------------------------
# Baseline 3: BERT + single adapter (no domain adaptation)
# ---------------------------------------------------------------------------

class BERTPlusAdapter(nn.Module):
    """BERT with a single adapter stack (no adaptation mechanism)."""

    def __init__(
        self,
        bert_model_name: str,
        num_event_types: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        from daa.modules.bert import FrozenBERTEncoder
        self.bert = FrozenBERTEncoder(bert_model_name, cache_dir=cache_dir)
        L = self.bert.num_layers
        H = self.bert.hidden_size
        self.adapter = DomainAdapterStack(L, H, bottleneck_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(H, num_event_types)

    def _trigger_repr(
        self, hidden_states, trigger_idx: torch.Tensor
    ) -> torch.Tensor:
        last = hidden_states[-1]
        B, T, H = last.shape
        idx = trigger_idx.clamp(0, T - 1).unsqueeze(1).unsqueeze(2).expand(B, 1, H)
        return last.gather(1, idx).squeeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trigger_idx: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        all_hidden = self.bert(input_ids, attention_mask)
        adapted = [
            self.adapter.forward_layer(l, all_hidden[l + 1])
            for l in range(self.bert.num_layers)
        ]
        h = self.dropout(self._trigger_repr(adapted, trigger_idx))
        logits = self.classifier(h)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = event_classification_loss(logits, labels)
        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trigger_idx: torch.Tensor,
    ) -> torch.Tensor:
        out = self.forward(input_ids, attention_mask, trigger_idx)
        return out["logits"].argmax(dim=-1)

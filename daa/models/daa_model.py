"""
Full DAA (Domain-specific Adapter-based Adaptation) model.

Architecture (§3 of the ACL 2021 paper):

  1. Frozen BERT encoder  B
  2. Three adapter stacks (one per domain × one per BERT layer):
       A^s — source-domain private adapter
       A^t — target-domain private adapter
       A^j — joint adapter (used for classification)
  3. Event-detection head  h_T : R^H → R^{num_types}
  4. MLM prediction head   h_M : R^H → R^V  (shared across domains)
  5. Layer-wise domain adversarial (LDA) heads: L classifiers h^J_{d,l}
  6. Wasserstein estimator h_W

Forward pass logic (per mini-batch):
  - Source batch goes through B, then A^s and A^j at each layer
  - Target batch goes through B, then A^t and A^j at each layer
  - Classification uses the joint adapter output on source data
  - LDA uses joint-adapter bottleneck representations
  - Orthogonality loss uses private + joint adapter outputs
  - MLM uses joint + private combined outputs

Training uses alternating minimisation (§3.4):
  Step 1: update LDA heads and Wasserstein estimator
  Step 2: update adapters + task head (using selected source samples)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from daa.modules.bert import FrozenBERTEncoder
from daa.modules.adapter import DomainAdapterStack
from daa.modules.domain_adversarial import LayerWiseDomainAdversarial
from daa.modules.wasserstein import WassersteinEstimator, gradient_penalty, wasserstein_loss
from daa.modules.losses import (
    event_classification_loss,
    mlm_loss,
    orthogonality_loss,
)


class DAAModel(nn.Module):
    """Full DAA model for unsupervised domain adaptation for event detection.

    Args:
        bert_model_name: HuggingFace model identifier (e.g. 'bert-base-uncased').
        num_event_types: Number of output classes (34 for ACE-05, 2 for binary).
        bottleneck_dim:  Adapter bottleneck dimension c.
        cache_dir:       Optional HuggingFace cache directory.
        classifier_hidden: Hidden size for LDA domain classifiers.
        wasserstein_hidden: Hidden size for Wasserstein estimator.
    """

    def __init__(
        self,
        bert_model_name: str,
        num_event_types: int,
        bottleneck_dim: int = 64,
        cache_dir: Optional[str] = None,
        classifier_hidden: int = 128,
        wasserstein_hidden: int = 256,
    ):
        super().__init__()

        # ── Frozen BERT ───────────────────────────────────────────────────
        self.bert = FrozenBERTEncoder(bert_model_name, cache_dir=cache_dir)
        L = self.bert.num_layers
        H = self.bert.hidden_size

        # ── Three adapter stacks ──────────────────────────────────────────
        self.adapter_src = DomainAdapterStack(L, H, bottleneck_dim)
        self.adapter_tgt = DomainAdapterStack(L, H, bottleneck_dim)
        self.adapter_joint = DomainAdapterStack(L, H, bottleneck_dim)

        # ── Task head (event classification) ─────────────────────────────
        self.event_classifier = nn.Linear(H, num_event_types)

        # ── MLM head (shared across domains) ─────────────────────────────
        self.mlm_head = nn.Sequential(
            nn.Linear(H, H),
            nn.GELU(),
            nn.LayerNorm(H),
            nn.Linear(H, self.bert.vocab_size),
        )

        # ── LDA heads (L domain classifiers) ─────────────────────────────
        self.lda = LayerWiseDomainAdversarial(L, bottleneck_dim, classifier_hidden)

        # ── Wasserstein estimator ─────────────────────────────────────────
        self.wasserstein = WassersteinEstimator(H, wasserstein_hidden)

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        adapter_stack: DomainAdapterStack,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Pass inputs through BERT + one adapter stack at each layer.

        Returns:
            adapted_hidden_states: List of L adapted (B, T, H) tensors.
            bottlenecks:           List of L bottleneck (B, T, c) tensors.
        """
        all_hidden = self.bert(input_ids, attention_mask)
        # all_hidden[0] = embedding output, all_hidden[1..L] = transformer layers

        adapted = []
        bns = []
        h = all_hidden[0]  # start from embedding output
        for l in range(self.bert.num_layers):
            # BERT layer output
            bert_layer_out = all_hidden[l + 1]
            # Feed BERT layer output through adapter
            h_adapted, bn = adapter_stack.forward_layer_with_bottleneck(l, bert_layer_out)
            adapted.append(h_adapted)
            bns.append(bn)

        return adapted, bns

    def _trigger_repr(
        self,
        hidden_states: List[torch.Tensor],
        trigger_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the trigger-token representation from the last layer.

        Args:
            hidden_states: List of (B, T, H).
            trigger_idx:   (B,) — position of the trigger token.

        Returns:
            (B, H) — trigger-token representation.
        """
        last = hidden_states[-1]  # (B, T, H)
        B, T, H = last.shape
        idx = trigger_idx.clamp(0, T - 1).unsqueeze(1).unsqueeze(2).expand(B, 1, H)
        return last.gather(1, idx).squeeze(1)  # (B, H)

    # ------------------------------------------------------------------
    # Public forward methods
    # ------------------------------------------------------------------

    def forward_source(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trigger_idx: torch.Tensor,
    ) -> Dict[str, object]:
        """Forward pass for a source-domain batch.

        Returns a dict with:
            joint_repr      : (B, H) — joint adapter trigger representation
            private_repr    : (B, H) — source adapter trigger representation
            joint_all_layers: List of (B, T, H) per-layer joint outputs
            private_bns     : List of (B, T, c) private bottlenecks (unused)
            joint_bns       : List of (B, T, c) joint bottlenecks for LDA
        """
        # Joint adapter
        joint_adapted, joint_bns = self._encode(input_ids, attention_mask, self.adapter_joint)
        # Source-private adapter
        src_adapted, src_bns = self._encode(input_ids, attention_mask, self.adapter_src)

        joint_repr = self._trigger_repr(joint_adapted, trigger_idx)
        private_repr = self._trigger_repr(src_adapted, trigger_idx)

        return dict(
            joint_repr=joint_repr,
            private_repr=private_repr,
            joint_all_layers=joint_adapted,
            joint_bns=joint_bns,
            private_bns=src_bns,
        )

    def forward_target(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trigger_idx: torch.Tensor,
    ) -> Dict[str, object]:
        """Forward pass for a target-domain batch."""
        joint_adapted, joint_bns = self._encode(input_ids, attention_mask, self.adapter_joint)
        tgt_adapted, tgt_bns = self._encode(input_ids, attention_mask, self.adapter_tgt)

        joint_repr = self._trigger_repr(joint_adapted, trigger_idx)
        private_repr = self._trigger_repr(tgt_adapted, trigger_idx)

        return dict(
            joint_repr=joint_repr,
            private_repr=private_repr,
            joint_all_layers=joint_adapted,
            joint_bns=joint_bns,
            private_bns=tgt_bns,
        )

    # ------------------------------------------------------------------
    # Loss computation methods (used by the trainer)
    # ------------------------------------------------------------------

    def compute_task_loss(
        self,
        src_joint_repr: torch.Tensor,
        src_labels: torch.Tensor,
    ) -> torch.Tensor:
        """L_task — supervised event classification on source data."""
        logits = self.event_classifier(src_joint_repr)
        return event_classification_loss(logits, src_labels)

    def compute_lda_loss(
        self,
        src_joint_bns: List[torch.Tensor],
        tgt_joint_bns: List[torch.Tensor],
        src_domain_labels: torch.Tensor,
        tgt_domain_labels: torch.Tensor,
    ) -> torch.Tensor:
        """L_D — layer-wise domain adversarial loss.

        For the *classifier update* step: maximise over LDA head parameters.
        For the *adapter update* step: called with reversed domain labels.
        """
        # Combine source + target bottlenecks per layer
        combined_bns = [
            torch.cat([s, t], dim=0)
            for s, t in zip(src_joint_bns, tgt_joint_bns)
        ]
        combined_domain = torch.cat([src_domain_labels, tgt_domain_labels], dim=0)
        return self.lda(combined_bns, combined_domain)

    def compute_wasserstein_loss(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        src_trigger_idx: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        tgt_attention_mask: torch.Tensor,
        tgt_trigger_idx: torch.Tensor,
        gp_lambda: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """L_W — Wasserstein distance estimation loss + gradient penalty.

        Returns:
            w_loss:  Wasserstein estimate (to be maximised w.r.t. estimator).
            gp_loss: Gradient penalty.
        """
        src_adapted, _ = self._encode(src_input_ids, src_attention_mask, self.adapter_src)
        tgt_adapted, _ = self._encode(tgt_input_ids, tgt_attention_mask, self.adapter_tgt)

        src_repr = self._trigger_repr(src_adapted, src_trigger_idx)
        tgt_repr = self._trigger_repr(tgt_adapted, tgt_trigger_idx)

        w_loss = wasserstein_loss(self.wasserstein, src_repr, tgt_repr)
        gp_loss = gradient_penalty(self.wasserstein, src_repr, tgt_repr)
        return w_loss, gp_loss

    def compute_orthogonality_loss(
        self,
        src_joint_repr: torch.Tensor,
        src_private_repr: torch.Tensor,
        tgt_joint_repr: torch.Tensor,
        tgt_private_repr: torch.Tensor,
    ) -> torch.Tensor:
        """L_S — Frobenius orthogonality constraint."""
        return orthogonality_loss(
            src_joint_repr, src_private_repr,
            tgt_joint_repr, tgt_private_repr,
        )

    def compute_mlm_loss(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        src_masked_input_ids: torch.Tensor,
        src_mlm_labels: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        tgt_attention_mask: torch.Tensor,
        tgt_masked_input_ids: torch.Tensor,
        tgt_mlm_labels: torch.Tensor,
    ) -> torch.Tensor:
        """L_M — MLM auxiliary loss using joint + private adapter combined.

        h_M(A^j(x) + A^d(x)) where d matches the input domain.
        """
        # Source MLM: A^j + A^s
        src_joint_adapted, _ = self._encode(src_masked_input_ids, src_attention_mask, self.adapter_joint)
        src_private_adapted, _ = self._encode(src_masked_input_ids, src_attention_mask, self.adapter_src)
        src_combined = src_joint_adapted[-1] + src_private_adapted[-1]  # (B, T, H)
        src_mlm_logits = self.mlm_head(src_combined)

        # Target MLM: A^j + A^t
        tgt_joint_adapted, _ = self._encode(tgt_masked_input_ids, tgt_attention_mask, self.adapter_joint)
        tgt_private_adapted, _ = self._encode(tgt_masked_input_ids, tgt_attention_mask, self.adapter_tgt)
        tgt_combined = tgt_joint_adapted[-1] + tgt_private_adapted[-1]  # (B, T, H)
        tgt_mlm_logits = self.mlm_head(tgt_combined)

        src_loss = mlm_loss(src_mlm_logits, src_mlm_labels)
        tgt_loss = mlm_loss(tgt_mlm_logits, tgt_mlm_labels)
        return src_loss + tgt_loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trigger_idx: torch.Tensor,
    ) -> torch.Tensor:
        """At test time, use only the joint adapter for prediction.

        Returns:
            (B,) — predicted class IDs.
        """
        joint_adapted, _ = self._encode(input_ids, attention_mask, self.adapter_joint)
        joint_repr = self._trigger_repr(joint_adapted, trigger_idx)
        logits = self.event_classifier(joint_repr)
        return logits.argmax(dim=-1)

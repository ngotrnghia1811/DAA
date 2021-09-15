"""
Trainer for the DAA model.

Implements the alternating minimisation procedure described in §3.4:

  Step 1 (repeated k times):
    - Fix adapters + task head
    - Update LDA domain classifiers  (maximise L_D)
    - Update Wasserstein estimator   (maximise L_W − λ_GP · L_GP)
    - Select source samples with lowest Wasserstein scores

  Step 2 (once):
    - Fix LDA heads and Wasserstein estimator
    - Update adapters + task head using:
        L_total = L_task + λ_d·L_D(reversed) + λ_w·L_W(minimise) + λ_s·L_S + λ_m·L_M

Evaluation at each epoch: standard P/R/F1 over test set.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer

from daa.config import Config
from daa.data import EventDataset, UnlabeledDataset
from daa.models.daa_model import DAAModel
from daa.scorer import compute_prf
from daa.util import apply_mlm_mask
from daa.vocabs import Vocab
from daa.modules.wasserstein import select_source_samples

logger = logging.getLogger(__name__)


class DAATrainer:
    """Trains a DAAModel using alternating minimisation."""

    def __init__(
        self,
        model: DAAModel,
        config: Config,
        event_vocab: Vocab,
        device: torch.device,
        output_dir: str,
    ):
        self.model = model
        self.config = config
        self.event_vocab = event_vocab
        self.device = device
        self.output_dir = output_dir

        # ── Separate parameter groups ────────────────────────────────────
        # Group 1: adversarial / estimator heads (Step 1)
        disc_params = (
            list(model.lda.parameters()) +
            list(model.wasserstein.parameters())
        )
        # Group 2: adapters + task/MLM heads (Step 2)
        adapter_params = (
            list(model.adapter_src.parameters()) +
            list(model.adapter_tgt.parameters()) +
            list(model.adapter_joint.parameters()) +
            list(model.event_classifier.parameters()) +
            list(model.mlm_head.parameters())
        )

        self.opt_disc = AdamW(disc_params, lr=config.head_lr, weight_decay=config.weight_decay)
        self.opt_adapter = AdamW(adapter_params, lr=config.adapter_lr, weight_decay=config.weight_decay)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        src_loader: DataLoader,
        tgt_loader: DataLoader,
        test_loader: DataLoader,
        tokenizer: BertTokenizer,
    ) -> None:
        """Run full training loop."""
        best_f1 = 0.0
        best_model_path = os.path.join(self.output_dir, "best_model.pt")
        log_path = os.path.join(self.output_dir, "train_log.jsonl")

        for epoch in range(self.config.max_epoch):
            train_losses = self._train_epoch(src_loader, tgt_loader, tokenizer)
            metrics = self._evaluate(test_loader)

            record = {"epoch": epoch, **train_losses, **metrics}
            logger.info(json.dumps(record))
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(
                    {"model_state": self.model.state_dict(), "epoch": epoch},
                    best_model_path,
                )
                logger.info(f"New best model saved (F1={best_f1:.4f})")

        logger.info(f"Training complete. Best F1 = {best_f1:.4f}")

    def _train_epoch(
        self,
        src_loader: DataLoader,
        tgt_loader: DataLoader,
        tokenizer: BertTokenizer,
    ) -> Dict[str, float]:
        """One training epoch."""
        self.model.train()
        tgt_iter = iter(tgt_loader)

        total_task = total_domain = total_wass = total_sim = total_mlm = 0.0
        num_batches = 0

        for src_batch in src_loader:
            # Fetch target batch (cycle if exhausted)
            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_batch = next(tgt_iter)

            src_batch = {k: v.to(self.device) for k, v in src_batch.items()}
            tgt_batch = {k: v.to(self.device) for k, v in tgt_batch.items()}

            # ── Step 1: update discriminators k times ─────────────────
            for _ in range(self.config.alt_min_steps_k):
                step1_losses = self._step1_update_discriminators(src_batch, tgt_batch)

            # ── Step 2: update adapters ───────────────────────────────
            # Select transferable source samples
            sel_idx = self._select_source_samples(src_batch)
            step2_losses = self._step2_update_adapters(
                src_batch, tgt_batch, sel_idx, tokenizer
            )

            total_task += step2_losses["task"]
            total_domain += step2_losses["domain"]
            total_wass += step2_losses["wasserstein"]
            total_sim += step2_losses["similarity"]
            total_mlm += step2_losses["mlm"]
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "loss_task": total_task / n,
            "loss_domain": total_domain / n,
            "loss_wasserstein": total_wass / n,
            "loss_similarity": total_sim / n,
            "loss_mlm": total_mlm / n,
        }

    @torch.no_grad()
    def _select_source_samples(self, src_batch: dict) -> torch.Tensor:
        """Use the Wasserstein estimator to rank and select source samples."""
        src_adapted, _ = self.model._encode(
            src_batch["input_ids"],
            src_batch["attention_mask"],
            self.model.adapter_src,
        )
        src_repr = self.model._trigger_repr(src_adapted, src_batch["trigger_idx"])
        return select_source_samples(
            self.model.wasserstein, src_repr, self.config.data_selection_ratio
        )

    def _step1_update_discriminators(
        self, src_batch: dict, tgt_batch: dict
    ) -> Dict[str, float]:
        """Maximise L_D (domain classifiers) and L_W (Wasserstein estimator)."""
        self.opt_disc.zero_grad()

        # Compute LDA loss — use actual domain labels (0=src, 1=tgt)
        src_out = self.model.forward_source(
            src_batch["input_ids"],
            src_batch["attention_mask"],
            src_batch["trigger_idx"],
        )
        tgt_out = self.model.forward_target(
            tgt_batch["input_ids"],
            tgt_batch["attention_mask"],
            tgt_batch["trigger_idx"],
        )

        src_domain = src_batch["domain_labels"]
        tgt_domain = tgt_batch["domain_labels"]

        lda_loss = self.model.compute_lda_loss(
            src_out["joint_bns"], tgt_out["joint_bns"],
            src_domain, tgt_domain,
        )
        # Maximise → negate
        (-lda_loss).backward(retain_graph=True)

        # Wasserstein estimation
        w_loss, gp_loss = self.model.compute_wasserstein_loss(
            src_batch["input_ids"], src_batch["attention_mask"], src_batch["trigger_idx"],
            tgt_batch["input_ids"], tgt_batch["attention_mask"], tgt_batch["trigger_idx"],
            gp_lambda=self.config.wasserstein_gp_lambda,
        )
        w_total = -(w_loss - self.config.wasserstein_gp_lambda * gp_loss)
        w_total.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.model.lda.parameters()) + list(self.model.wasserstein.parameters()),
            self.config.grad_clipping,
        )
        self.opt_disc.step()

        return {"lda_loss": lda_loss.item(), "w_loss": w_loss.item()}

    def _step2_update_adapters(
        self,
        src_batch: dict,
        tgt_batch: dict,
        selected_src_idx: torch.Tensor,
        tokenizer: BertTokenizer,
    ) -> Dict[str, float]:
        """Minimise L_total over adapter + task head parameters."""
        self.opt_adapter.zero_grad()

        # Select subset of source batch
        sel_input_ids = src_batch["input_ids"][selected_src_idx]
        sel_attention_mask = src_batch["attention_mask"][selected_src_idx]
        sel_trigger_idx = src_batch["trigger_idx"][selected_src_idx]
        sel_labels = src_batch["labels"][selected_src_idx]
        sel_domain = src_batch["domain_labels"][selected_src_idx]

        # Forward passes
        src_out = self.model.forward_source(
            sel_input_ids, sel_attention_mask, sel_trigger_idx,
        )
        tgt_out = self.model.forward_target(
            tgt_batch["input_ids"],
            tgt_batch["attention_mask"],
            tgt_batch["trigger_idx"],
        )

        # L_task
        task_loss = self.model.compute_task_loss(src_out["joint_repr"], sel_labels)

        # L_D (reversed domain labels to fool classifiers)
        reversed_src_domain = 1 - sel_domain
        reversed_tgt_domain = 1 - tgt_batch["domain_labels"]
        lda_loss = self.model.compute_lda_loss(
            src_out["joint_bns"], tgt_out["joint_bns"],
            reversed_src_domain, reversed_tgt_domain,
        )

        # L_W (minimise Wasserstein distance between distributions)
        w_loss, gp_loss = self.model.compute_wasserstein_loss(
            sel_input_ids, sel_attention_mask, sel_trigger_idx,
            tgt_batch["input_ids"], tgt_batch["attention_mask"], tgt_batch["trigger_idx"],
            gp_lambda=self.config.wasserstein_gp_lambda,
        )

        # L_S (orthogonality)
        sim_loss = self.model.compute_orthogonality_loss(
            src_out["joint_repr"], src_out["private_repr"],
            tgt_out["joint_repr"], tgt_out["private_repr"],
        )

        # L_M (MLM)
        src_masked_ids, src_mlm_labels = apply_mlm_mask(
            sel_input_ids, self.model.bert.vocab_size,
            tokenizer.mask_token_id, tokenizer.pad_token_id or 0,
            mask_prob=self.config.mlm_mask_prob,
        )
        tgt_masked_ids, tgt_mlm_labels = apply_mlm_mask(
            tgt_batch["input_ids"], self.model.bert.vocab_size,
            tokenizer.mask_token_id, tokenizer.pad_token_id or 0,
            mask_prob=self.config.mlm_mask_prob,
        )
        mlm_loss_val = self.model.compute_mlm_loss(
            sel_input_ids, sel_attention_mask, src_masked_ids, src_mlm_labels,
            tgt_batch["input_ids"], tgt_batch["attention_mask"], tgt_masked_ids, tgt_mlm_labels,
        )

        # Total loss
        total = (
            task_loss
            + self.config.lambda_domain * lda_loss
            + self.config.lambda_wasserstein * w_loss
            + self.config.lambda_similarity * sim_loss
            + self.config.lambda_mlm * mlm_loss_val
        )
        total.backward()

        torch.nn.utils.clip_grad_norm_(
            (list(self.model.adapter_src.parameters()) +
             list(self.model.adapter_tgt.parameters()) +
             list(self.model.adapter_joint.parameters()) +
             list(self.model.event_classifier.parameters()) +
             list(self.model.mlm_head.parameters())),
            self.config.grad_clipping,
        )
        self.opt_adapter.step()

        return {
            "task": task_loss.item(),
            "domain": lda_loss.item(),
            "wasserstein": w_loss.item(),
            "similarity": sim_loss.item(),
            "mlm": mlm_loss_val.item(),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on the test/dev set."""
        self.model.eval()
        gold_all: List[int] = []
        pred_all: List[int] = []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            preds = self.model.predict(
                batch["input_ids"],
                batch["attention_mask"],
                batch["trigger_idx"],
            )
            gold_all.extend(batch["labels"].cpu().tolist())
            pred_all.extend(preds.cpu().tolist())

        none_id = self.event_vocab.token2id("None", default=0)
        metrics = compute_prf(gold_all, pred_all, none_label_id=none_id)
        return metrics

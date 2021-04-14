"""
Configuration for DAA models.

Each experiment corresponds to a JSON config file.  See configs/ for examples.
"""
import copy
import json
import os
from typing import Any, Dict, List, Optional


class Config:
    """Hyper-parameters and paths for a single DAA experiment."""

    def __init__(self, **kwargs):
        # ── BERT base ──────────────────────────────────────────────────────
        self.bert_model_name: str = kwargs.pop("bert_model_name", "bert-base-uncased")
        self.bert_cache_dir: Optional[str] = kwargs.pop("bert_cache_dir", None)
        self.bert_dropout: float = kwargs.pop("bert_dropout", 0.1)

        # ── Adapter architecture ───────────────────────────────────────────
        # Bottleneck dimension c (c << d_model).
        self.adapter_bottleneck_dim: int = kwargs.pop("adapter_bottleneck_dim", 64)

        # ── Task ──────────────────────────────────────────────────────────
        # 'event_detection'  – multi-class (ACE-05, 34 types incl. None)
        # 'event_identification' – binary (TimeBank, LitBank)
        self.task: str = kwargs.pop("task", "event_detection")
        self.num_event_types: int = kwargs.pop("num_event_types", 34)

        # ── Datasets ─────────────────────────────────────────────────────
        self.source_train_file: Optional[str] = kwargs.pop("source_train_file", None)
        self.source_dev_file: Optional[str] = kwargs.pop("source_dev_file", None)
        self.target_train_file: Optional[str] = kwargs.pop("target_train_file", None)
        self.target_dev_file: Optional[str] = kwargs.pop("target_dev_file", None)
        self.test_file: Optional[str] = kwargs.pop("test_file", None)
        self.log_path: Optional[str] = kwargs.pop("log_path", "output")

        # ── Training ──────────────────────────────────────────────────────
        # Paper uses 150 total per step: 90 source + 60 target
        self.src_batch_size: int = kwargs.pop("src_batch_size", 90)
        self.tgt_batch_size: int = kwargs.pop("tgt_batch_size", 60)
        self.eval_batch_size: int = kwargs.pop("eval_batch_size", 32)
        self.max_epoch: int = kwargs.pop("max_epoch", 50)
        self.max_seq_length: int = kwargs.pop("max_seq_length", 128)
        self.warmup_epoch: int = kwargs.pop("warmup_epoch", 5)
        self.accumulate_step: int = kwargs.pop("accumulate_step", 1)
        self.grad_clipping: float = kwargs.pop("grad_clipping", 5.0)

        # ── Optimizers ────────────────────────────────────────────────────
        self.adapter_lr: float = kwargs.pop("adapter_lr", 1e-4)
        self.head_lr: float = kwargs.pop("head_lr", 1e-3)
        self.weight_decay: float = kwargs.pop("weight_decay", 1e-4)

        # ── Alternating minimization ───────────────────────────────────────
        # k: number of discriminator/estimator update steps per adapter step.
        self.alt_min_steps_k: int = kwargs.pop("alt_min_steps_k", 5)

        # ── Loss weights ──────────────────────────────────────────────────
        # L_total = L_task + λ_d*L_D + λ_w*L_W + λ_s*L_S + λ_m*L_M
        self.lambda_domain: float = kwargs.pop("lambda_domain", 1.0)
        self.lambda_wasserstein: float = kwargs.pop("lambda_wasserstein", 1.0)
        self.lambda_similarity: float = kwargs.pop("lambda_similarity", 0.5)
        self.lambda_mlm: float = kwargs.pop("lambda_mlm", 1.0)

        # ── Wasserstein gradient penalty ──────────────────────────────────
        self.wasserstein_gp_lambda: float = kwargs.pop("wasserstein_gp_lambda", 10.0)

        # ── Data selection ────────────────────────────────────────────────
        # Fraction of source batch selected for joint adapter DANN training.
        self.data_selection_ratio: float = kwargs.pop("data_selection_ratio", 0.7)

        # ── MLM ───────────────────────────────────────────────────────────
        self.mlm_mask_prob: float = kwargs.pop("mlm_mask_prob", 0.15)

        # ── Device ───────────────────────────────────────────────────────
        self.use_gpu: bool = kwargs.pop("use_gpu", True)
        self.gpu_device: int = kwargs.pop("gpu_device", -1)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        cfg = cls()
        for k, v in d.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_json_file(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        print(f"Config saved to {path}")

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"Config({fields})"

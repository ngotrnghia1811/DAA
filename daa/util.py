"""
Miscellaneous utilities.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(use_gpu: bool, gpu_device: int = -1) -> torch.device:
    """Return the appropriate torch device."""
    if use_gpu and torch.cuda.is_available():
        if gpu_device >= 0:
            return torch.device(f"cuda:{gpu_device}")
        return torch.device("cuda")
    return torch.device("cpu")


def create_output_dir(log_path: str) -> str:
    """Create a timestamped output directory."""
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = os.path.join(log_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def configure_logging(log_file: str) -> None:
    """Set up console + file logging."""
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def save_predictions(
    path: str,
    sent_ids: List[str],
    gold_labels: List[int],
    pred_labels: List[int],
    id2label: Dict[int, str],
) -> None:
    """Write predictions to a JSON-lines file for later analysis."""
    with open(path, "w", encoding="utf-8") as fh:
        for sid, gold, pred in zip(sent_ids, gold_labels, pred_labels):
            fh.write(
                json.dumps(
                    {
                        "sent_id": sid,
                        "gold": id2label[gold],
                        "pred": id2label[pred],
                    }
                )
                + "\n"
            )


def apply_mlm_mask(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    pad_token_id: int,
    mask_prob: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random masking for MLM loss.

    Returns:
        masked_input_ids: input with some tokens replaced by [MASK].
        mlm_labels: target IDs (-100 for unmasked positions).
    """
    labels = input_ids.clone()
    # Do not mask padding tokens
    probability_matrix = torch.full(labels.shape, mask_prob, device=input_ids.device)
    padding_mask = input_ids.eq(pad_token_id)
    probability_matrix.masked_fill_(padding_mask, 0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # 80 % → [MASK], 10 % → random, 10 % → unchanged
    indices_replaced = torch.bernoulli(
        torch.full(labels.shape, 0.8, device=input_ids.device)
    ).bool() & masked_indices
    input_ids_out = input_ids.clone()
    input_ids_out[indices_replaced] = mask_token_id

    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5, device=input_ids.device)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=input_ids.device)
    input_ids_out[indices_random] = random_words[indices_random]

    return input_ids_out, labels

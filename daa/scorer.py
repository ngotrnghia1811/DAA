"""
Evaluation metrics (Precision / Recall / F1) for event detection.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


def compute_prf(
    gold_labels: List[int],
    pred_labels: List[int],
    none_label_id: int = 0,
) -> Dict[str, float]:
    """Micro-averaged Precision, Recall, F1 over non-None event types.

    Args:
        gold_labels: Ground-truth label IDs (one per token candidate).
        pred_labels: Predicted label IDs.
        none_label_id: The int ID of the "None" (negative) class.

    Returns:
        dict with keys "precision", "recall", "f1".
    """
    tp = fp = fn = 0
    for gold, pred in zip(gold_labels, pred_labels):
        is_gold_event = gold != none_label_id
        is_pred_event = pred != none_label_id
        if is_gold_event and is_pred_event and gold == pred:
            tp += 1
        elif is_pred_event and (not is_gold_event or gold != pred):
            fp += 1
        elif is_gold_event and (not is_pred_event or gold != pred):
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_identification_prf(
    gold_labels: List[int],
    pred_labels: List[int],
    none_label_id: int = 0,
) -> Dict[str, float]:
    """Binary P/R/F for event identification tasks (TimeBank / LitBank)."""
    return compute_prf(gold_labels, pred_labels, none_label_id)

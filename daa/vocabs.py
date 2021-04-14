"""
Vocabulary utilities for DAA event detection.

Builds event-type → index mappings from the ACE-05 guideline or from
the training data for binary datasets (TimeBank / LitBank).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# ACE-05 event types (33 types + "None" = 34 classes)
# ---------------------------------------------------------------------------
ACE05_EVENT_TYPES: List[str] = [
    "None",
    "Be-Born", "Marry", "Divorce", "Injure", "Die",
    "Transport", "Transfer-Ownership", "Transfer-Money",
    "Start-Org", "Merge-Org", "Declare-Bankruptcy", "End-Org",
    "Attack", "Demonstrate", "Meet", "Phone-Write",
    "Start-Position", "End-Position", "Nominate", "Elect",
    "Arrest-Jail", "Release-Parole", "Charge-Indict", "Trial-Hearing",
    "Convict", "Sentence", "Execute", "Acquit", "Pardon",
    "Appeal", "Fine", "Extradite",
    "Sue",
]

# Binary datasets
BINARY_EVENT_TYPES: List[str] = ["None", "Event"]


class Vocab:
    """Bidirectional string ↔ int mapping."""

    def __init__(self, tokens: Iterable[str]):
        self._token2id: Dict[str, int] = {}
        self._id2token: Dict[int, str] = {}
        for tok in tokens:
            if tok not in self._token2id:
                idx = len(self._token2id)
                self._token2id[tok] = idx
                self._id2token[idx] = tok

    def __len__(self) -> int:
        return len(self._token2id)

    def __contains__(self, token: str) -> bool:
        return token in self._token2id

    def token2id(self, token: str, default: Optional[int] = None) -> Optional[int]:
        return self._token2id.get(token, default)

    def id2token(self, idx: int) -> str:
        return self._id2token[idx]

    @property
    def tokens(self) -> List[str]:
        return [self._id2token[i] for i in range(len(self._id2token))]


def build_event_vocab(task: str = "event_detection") -> Vocab:
    """Return the Vocab for a given task."""
    if task == "event_detection":
        return Vocab(ACE05_EVENT_TYPES)
    elif task == "event_identification":
        return Vocab(BINARY_EVENT_TYPES)
    else:
        raise ValueError(f"Unknown task: {task!r}")


def build_event_vocab_from_data(labels: Iterable[str]) -> Vocab:
    """Build a Vocab dynamically from a label set (keeps 'None' first)."""
    unique = ["None"] + sorted({l for l in labels if l != "None"})
    return Vocab(unique)

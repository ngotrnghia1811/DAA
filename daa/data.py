"""
Dataset classes for the DAA project.

Both the source (labeled) and target (unlabeled) domains are represented
as PyTorch Datasets.  A combined loader for alternating-minimization
training is also provided.

Input JSON-lines format (one sentence per line):
    {
        "sent_id": "doc-001-0",
        "tokens": ["The", "police", "fired", ...],
        "event_mentions": [
            {"trigger": {"start": 2, "end": 3, "text": "fired"},
             "event_type": "Attack"}
        ]
    }

For unlabeled target data, "event_mentions" may be omitted or empty.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from daa.vocabs import Vocab


# ---------------------------------------------------------------------------
# Single instance
# ---------------------------------------------------------------------------
class EventInstance:
    """One event mention with its trigger position and label."""

    __slots__ = (
        "sent_id", "tokens", "trigger_start", "trigger_end",
        "event_type", "domain_label",
        "input_ids", "attention_mask", "trigger_token_idx", "label_id",
    )

    def __init__(
        self,
        sent_id: str,
        tokens: List[str],
        trigger_start: int,
        trigger_end: int,
        event_type: str,
        domain_label: int,
    ):
        self.sent_id = sent_id
        self.tokens = tokens
        self.trigger_start = trigger_start
        self.trigger_end = trigger_end
        self.event_type = event_type
        self.domain_label = domain_label  # 0=source, 1=target

        # Filled by numberize()
        self.input_ids: Optional[List[int]] = None
        self.attention_mask: Optional[List[int]] = None
        self.trigger_token_idx: Optional[int] = None
        self.label_id: Optional[int] = None

    def numberize(
        self,
        tokenizer: PreTrainedTokenizer,
        event_vocab: Vocab,
        max_seq_length: int = 128,
    ) -> None:
        """Tokenize and encode the instance."""
        # Build token list with CLS/SEP
        word_to_piece: List[int] = []
        pieces: List[str] = ["[CLS]"]

        for i, word in enumerate(self.tokens):
            word_to_piece.append(len(pieces))
            word_pieces = tokenizer.tokenize(word)
            if not word_pieces:
                word_pieces = ["[UNK]"]
            pieces.extend(word_pieces)

        pieces.append("[SEP]")

        # Truncate if necessary (keep CLS and SEP)
        if len(pieces) > max_seq_length:
            pieces = pieces[: max_seq_length - 1] + ["[SEP]"]

        ids = tokenizer.convert_tokens_to_ids(pieces)
        mask = [1] * len(ids)

        # Pad
        pad_len = max_seq_length - len(ids)
        ids += [tokenizer.pad_token_id or 0] * pad_len
        mask += [0] * pad_len

        # Trigger token index (first sub-token of trigger word)
        trigger_idx = word_to_piece[self.trigger_start]
        if trigger_idx >= max_seq_length:
            trigger_idx = max_seq_length - 1

        self.input_ids = ids
        self.attention_mask = mask
        self.trigger_token_idx = trigger_idx
        self.label_id = event_vocab.token2id(self.event_type, default=0)


# ---------------------------------------------------------------------------
# Event Dataset (labeled, for source domain)
# ---------------------------------------------------------------------------
class EventDataset(Dataset):
    """Labeled event-mention dataset."""

    def __init__(
        self,
        path: str,
        domain_label: int,
        gpu: bool = False,
    ):
        self.domain_label = domain_label
        self.gpu = gpu
        self.instances: List[EventInstance] = []
        self._load(path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tokens = obj["tokens"]
                sent_id = obj.get("sent_id", "")
                for mention in obj.get("event_mentions", []):
                    trigger = mention["trigger"]
                    inst = EventInstance(
                        sent_id=sent_id,
                        tokens=tokens,
                        trigger_start=trigger["start"],
                        trigger_end=trigger["end"],
                        event_type=mention.get("event_type", "None"),
                        domain_label=self.domain_label,
                    )
                    self.instances.append(inst)

    def numberize(
        self,
        tokenizer: PreTrainedTokenizer,
        event_vocab: Vocab,
        max_seq_length: int = 128,
    ) -> None:
        for inst in self.instances:
            inst.numberize(tokenizer, event_vocab, max_seq_length)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> EventInstance:
        return self.instances[idx]

    def collate_fn(self, batch: List[EventInstance]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([i.input_ids for i in batch], dtype=torch.long)
        attention_mask = torch.tensor([i.attention_mask for i in batch], dtype=torch.long)
        trigger_idx = torch.tensor([i.trigger_token_idx for i in batch], dtype=torch.long)
        labels = torch.tensor([i.label_id for i in batch], dtype=torch.long)
        domain_labels = torch.tensor([i.domain_label for i in batch], dtype=torch.long)

        if self.gpu and torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            trigger_idx = trigger_idx.cuda()
            labels = labels.cuda()
            domain_labels = domain_labels.cuda()

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            trigger_idx=trigger_idx,
            labels=labels,
            domain_labels=domain_labels,
        )


# ---------------------------------------------------------------------------
# Unlabeled Dataset (for target domain — no event labels)
# ---------------------------------------------------------------------------
class UnlabeledDataset(Dataset):
    """Unlabeled sentence dataset for the target domain.

    All tokens in every sentence window (centered around a candidate position)
    are encoded.  In practice, at inference time real trigger candidates come
    from the test set; during training we just use sentence-level windows.
    """

    def __init__(
        self,
        path: str,
        domain_label: int = 1,
        gpu: bool = False,
        window_size: int = 128,
    ):
        self.domain_label = domain_label
        self.gpu = gpu
        self.window_size = window_size
        self.instances: List[EventInstance] = []
        self._load(path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tokens = obj["tokens"]
                sent_id = obj.get("sent_id", "")
                # For unsupervised target: create one fake instance per token
                # as a potential trigger candidate (label = None / 0).
                for i in range(len(tokens)):
                    inst = EventInstance(
                        sent_id=sent_id,
                        tokens=tokens,
                        trigger_start=i,
                        trigger_end=i + 1,
                        event_type="None",
                        domain_label=self.domain_label,
                    )
                    self.instances.append(inst)

    def numberize(
        self,
        tokenizer: PreTrainedTokenizer,
        event_vocab: Vocab,
        max_seq_length: int = 128,
    ) -> None:
        for inst in self.instances:
            inst.numberize(tokenizer, event_vocab, max_seq_length)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> EventInstance:
        return self.instances[idx]

    def collate_fn(self, batch: List[EventInstance]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([i.input_ids for i in batch], dtype=torch.long)
        attention_mask = torch.tensor([i.attention_mask for i in batch], dtype=torch.long)
        trigger_idx = torch.tensor([i.trigger_token_idx for i in batch], dtype=torch.long)
        domain_labels = torch.tensor([i.domain_label for i in batch], dtype=torch.long)

        if self.gpu and torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            trigger_idx = trigger_idx.cuda()
            domain_labels = domain_labels.cuda()

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            trigger_idx=trigger_idx,
            domain_labels=domain_labels,
        )

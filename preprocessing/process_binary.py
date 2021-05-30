"""
Preprocessing for TimeBank and LitBank datasets.

Both datasets use binary event labels (Event / None).

TimeBank format assumed: standard XML annotation package.
LitBank format assumed: LitBank-events TSV or JSON annotations.

This script converts either to the unified DAA JSON-lines format.

Usage:
    # TimeBank
    python preprocessing/process_binary.py \
        --dataset timebank \
        --input_dir /path/to/timebank/data \
        --output_dir data/timebank

    # LitBank
    python preprocessing/process_binary.py \
        --dataset litbank \
        --input_dir /path/to/litbank/events \
        --output_dir data/litbank
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def _write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def process_timebank(input_dir: str, output_dir: str) -> None:
    """Convert TimeBank token+label TSV to DAA JSON-lines.

    Expected TSV format (one token per line, blank line = sentence boundary):
        token<TAB>BIO_label
    where BIO_label is B-EVENT, I-EVENT, or O.
    """
    for split in ["train", "dev", "test"]:
        tsv_path = os.path.join(input_dir, f"{split}.tsv")
        if not os.path.exists(tsv_path):
            print(f"[skip] {tsv_path} not found")
            continue

        records = []
        tokens: List[str] = []
        labels: List[str] = []
        sent_idx = 0

        with open(tsv_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    if tokens:
                        mentions = _extract_bio_mentions(tokens, labels, "Event")
                        records.append({
                            "sent_id": f"timebank-{split}-{sent_idx}",
                            "tokens": list(tokens),
                            "event_mentions": mentions,
                        })
                        sent_idx += 1
                        tokens, labels = [], []
                else:
                    parts = line.split("\t")
                    tokens.append(parts[0])
                    labels.append(parts[1] if len(parts) > 1 else "O")

        out_path = os.path.join(output_dir, f"{split}.jsonl")
        _write_jsonl(records, out_path)
        print(f"  Wrote {len(records)} sentences → {out_path}")

    # Unlabeled version of train for target
    train_path = os.path.join(output_dir, "train.jsonl")
    if os.path.exists(train_path):
        ul_path = os.path.join(output_dir, "train_unlabeled.jsonl")
        with open(train_path) as fi, open(ul_path, "w") as fo:
            for line in fi:
                obj = json.loads(line)
                obj["event_mentions"] = []
                fo.write(json.dumps(obj) + "\n")
        print(f"  Wrote unlabeled → {ul_path}")


def process_litbank(input_dir: str, output_dir: str) -> None:
    """Convert LitBank events TSV to DAA JSON-lines.

    Same BIO-TSV assumption as TimeBank above.
    """
    for split in ["train", "dev", "test"]:
        tsv_path = os.path.join(input_dir, f"{split}.tsv")
        if not os.path.exists(tsv_path):
            print(f"[skip] {tsv_path} not found")
            continue

        records = []
        tokens: List[str] = []
        labels: List[str] = []
        sent_idx = 0

        with open(tsv_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    if tokens:
                        mentions = _extract_bio_mentions(tokens, labels, "Event")
                        records.append({
                            "sent_id": f"litbank-{split}-{sent_idx}",
                            "tokens": list(tokens),
                            "event_mentions": mentions,
                        })
                        sent_idx += 1
                        tokens, labels = [], []
                else:
                    parts = line.split("\t")
                    tokens.append(parts[0])
                    labels.append(parts[1] if len(parts) > 1 else "O")

        out_path = os.path.join(output_dir, f"{split}.jsonl")
        _write_jsonl(records, out_path)
        print(f"  Wrote {len(records)} sentences → {out_path}")

    # Unlabeled
    train_path = os.path.join(output_dir, "train.jsonl")
    if os.path.exists(train_path):
        ul_path = os.path.join(output_dir, "train_unlabeled.jsonl")
        with open(train_path) as fi, open(ul_path, "w") as fo:
            for line in fi:
                obj = json.loads(line)
                obj["event_mentions"] = []
                fo.write(json.dumps(obj) + "\n")
        print(f"  Wrote unlabeled → {ul_path}")


def _extract_bio_mentions(
    tokens: List[str], labels: List[str], event_type: str
) -> List[Dict[str, Any]]:
    """Convert BIO token labels to event mention list."""
    mentions = []
    i = 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            start = i
            i += 1
            while i < len(labels) and labels[i].startswith("I-"):
                i += 1
            end = i
            mentions.append({
                "trigger": {
                    "start": start,
                    "end": end,
                    "text": " ".join(tokens[start:end]),
                },
                "event_type": event_type,
            })
        else:
            i += 1
    return mentions


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess binary event datasets.")
    parser.add_argument("--dataset", choices=["timebank", "litbank"], required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if args.dataset == "timebank":
        process_timebank(args.input_dir, args.output_dir)
    else:
        process_litbank(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

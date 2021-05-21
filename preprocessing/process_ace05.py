"""
Preprocessing for ACE-05 dataset → DAA JSON-lines format.

Converts ACE-05 annotation files into the unified format used by
EventDataset / UnlabeledDataset.

Output format (one JSON object per line):
    {
        "sent_id": "<doc_id>-<sent_idx>",
        "tokens": ["word1", "word2", ...],
        "event_mentions": [
            {
                "trigger": {"start": <int>, "end": <int>, "text": "<str>"},
                "event_type": "<str>"
            },
            ...
        ]
    }

Usage:
    python preprocessing/process_ace05.py \
        --input_dir /path/to/ace05/data/English \
        --output_dir data/ace05 \
        --split_dir ../DAA/_references  (optional, for official splits)

Note:
    This script assumes the ACE-05 corpus has been converted to a
    sentence-level format (e.g. via the OneIE preprocessing pipeline or
    DyGIE++ format).  Adapt the reader to your specific ACE-05 version.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helper: convert DyGIE++ / OneIE style to DAA format
# ---------------------------------------------------------------------------

def convert_oneie_style(src_path: str, dst_path: str, domain_label_filter: str = None) -> int:
    """Convert a OneIE-preprocessed JSON file to DAA JSON-lines.

    Args:
        src_path:           Input file (OneIE format: one JSON per line).
        dst_path:           Output file (DAA JSON-lines format).
        domain_label_filter: If given, only keep sentences from this domain.

    Returns:
        Number of event mentions written.
    """
    num_mentions = 0
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)

    with open(src_path, "r", encoding="utf-8") as fh_in, \
         open(dst_path, "w", encoding="utf-8") as fh_out:

        for line in fh_in:
            line = line.strip()
            if not line:
                continue
            obj: Dict[str, Any] = json.loads(line)

            tokens: List[str] = obj.get("tokens", obj.get("words", []))
            sent_id: str = obj.get("sent_id", obj.get("doc_id", "unknown"))

            # Gather event mentions
            mentions = []
            for em in obj.get("event_mentions", []):
                trigger = em.get("trigger", {})
                start = trigger.get("start", trigger.get("offset", [None])[0])
                end = trigger.get("end", start + 1 if start is not None else None)
                event_type = em.get("event_type", em.get("event_subtype", "None"))
                if start is None:
                    continue
                mentions.append({
                    "trigger": {"start": start, "end": end, "text": " ".join(tokens[start:end])},
                    "event_type": event_type,
                })
                num_mentions += 1

            out_obj = {
                "sent_id": sent_id,
                "tokens": tokens,
                "event_mentions": mentions,
            }
            fh_out.write(json.dumps(out_obj) + "\n")

    return num_mentions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ACE-05 for DAA.")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing OneIE-preprocessed JSON files.")
    parser.add_argument("--output_dir", default="data/ace05",
                        help="Output directory for DAA JSON-lines files.")
    parser.add_argument("--domains", nargs="+",
                        default=["bn+nw", "bc", "cts", "wl", "un"],
                        help="Domain sub-folders to process.")
    args = parser.parse_args()

    for domain in args.domains:
        for split in ["train", "dev", "test"]:
            src = os.path.join(args.input_dir, domain, f"{split}.json")
            dst = os.path.join(args.output_dir, domain, f"{split}.jsonl")
            if not os.path.exists(src):
                print(f"  [skip] {src} not found")
                continue
            n = convert_oneie_style(src, dst)
            print(f"  Wrote {n:6d} mentions → {dst}")

        # Also create an unlabeled version of train for target
        src = os.path.join(args.input_dir, domain, "train.json")
        dst_ul = os.path.join(args.output_dir, domain, "train_unlabeled.jsonl")
        if os.path.exists(src):
            # Strip event_mentions for unlabeled target version
            os.makedirs(os.path.dirname(dst_ul), exist_ok=True)
            with open(src) as fi, open(dst_ul, "w") as fo:
                for line in fi:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    obj["event_mentions"] = []
                    fo.write(json.dumps(obj) + "\n")
            print(f"  Wrote unlabeled → {dst_ul}")


if __name__ == "__main__":
    main()

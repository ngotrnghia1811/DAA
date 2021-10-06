"""
Evaluation script — run on a saved checkpoint.

Usage:
    python evaluate.py -c configs/ace05_bn2bc.json --checkpoint output/20240101_120000/best_model.pt
"""
import json
import logging
import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from daa.config import Config
from daa.data import EventDataset
from daa.models.daa_model import DAAModel
from daa.scorer import compute_prf
from daa.util import configure_logging, set_seed
from daa.vocabs import build_event_vocab


def main() -> None:
    parser = ArgumentParser(description="Evaluate a saved DAA checkpoint.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging(os.path.join(os.path.dirname(args.checkpoint), "eval.log"))
    logger = logging.getLogger(__name__)

    config = Config.from_json_file(args.config)
    set_seed(args.seed)

    if config.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpu_device}" if config.gpu_device >= 0 else "cuda")
    else:
        device = torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model_name, cache_dir=config.bert_cache_dir, do_lower_case=False
    )
    event_vocab = build_event_vocab(config.task)
    config.num_event_types = len(event_vocab)

    gpu = device.type == "cuda"
    test_set = EventDataset(config.test_file, domain_label=1, gpu=gpu)
    test_set.numberize(tokenizer, event_vocab, config.max_seq_length)
    test_loader = DataLoader(
        test_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=test_set.collate_fn,
    )

    # Load model
    model = DAAModel(
        bert_model_name=config.bert_model_name,
        num_event_types=config.num_event_types,
        bottleneck_dim=config.adapter_bottleneck_dim,
        cache_dir=config.bert_cache_dir,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Evaluate
    model.eval()
    gold_all, pred_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model.predict(
                batch["input_ids"], batch["attention_mask"], batch["trigger_idx"]
            )
            gold_all.extend(batch["labels"].cpu().tolist())
            pred_all.extend(preds.cpu().tolist())

    none_id = event_vocab.token2id("None", default=0)
    metrics = compute_prf(gold_all, pred_all, none_label_id=none_id)
    result_str = json.dumps(metrics, indent=2)
    logger.info(f"Results:\n{result_str}")
    print(result_str)


if __name__ == "__main__":
    main()

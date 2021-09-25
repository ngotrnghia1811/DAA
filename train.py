"""
Training entry-point for the DAA model.

Usage:
    python train.py -c configs/ace05_bn2bc.json

The script loads source (labeled) and target (unlabeled) datasets,
builds the DAAModel, and runs the alternating-minimisation training loop.
"""

import json
import logging
import os
import sys
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from daa.config import Config
from daa.data import EventDataset, UnlabeledDataset
from daa.models.daa_model import DAAModel
from daa.trainer import DAATrainer
from daa.util import configure_logging, create_output_dir, set_seed
from daa.vocabs import build_event_vocab


def main() -> None:
    parser = ArgumentParser(description="Train the DAA model.")
    parser.add_argument(
        "-c", "--config",
        default="configs/ace05_bn2bc.json",
        help="Path to JSON config file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────
    config = Config.from_json_file(args.config)
    set_seed(args.seed)

    # ── Output directory ─────────────────────────────────────────────────
    output_dir = create_output_dir(config.log_path or "output")
    log_file = os.path.join(output_dir, "train.log")
    configure_logging(log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output dir: {output_dir}")
    config.save(output_dir)

    # ── Device ───────────────────────────────────────────────────────────
    if config.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpu_device}" if config.gpu_device >= 0 else "cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model_name, cache_dir=config.bert_cache_dir, do_lower_case=False
    )

    # ── Vocabulary ───────────────────────────────────────────────────────
    event_vocab = build_event_vocab(config.task)
    config.num_event_types = len(event_vocab)
    logger.info(f"Event types: {len(event_vocab)}")

    # ── Datasets ─────────────────────────────────────────────────────────
    gpu = device.type == "cuda"

    src_train = EventDataset(config.source_train_file, domain_label=0, gpu=gpu)
    src_train.numberize(tokenizer, event_vocab, config.max_seq_length)

    src_dev = EventDataset(config.source_dev_file, domain_label=0, gpu=gpu)
    src_dev.numberize(tokenizer, event_vocab, config.max_seq_length)

    tgt_train = UnlabeledDataset(config.target_train_file, domain_label=1, gpu=gpu)
    tgt_train.numberize(tokenizer, event_vocab, config.max_seq_length)

    test_set = EventDataset(config.test_file, domain_label=1, gpu=gpu)
    test_set.numberize(tokenizer, event_vocab, config.max_seq_length)

    src_loader = DataLoader(
        src_train,
        batch_size=config.src_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=src_train.collate_fn,
    )
    tgt_loader = DataLoader(
        tgt_train,
        batch_size=config.tgt_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=tgt_train.collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=test_set.collate_fn,
    )

    logger.info(
        f"Source train: {len(src_train)} | Target train: {len(tgt_train)} | Test: {len(test_set)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = DAAModel(
        bert_model_name=config.bert_model_name,
        num_event_types=config.num_event_types,
        bottleneck_dim=config.adapter_bottleneck_dim,
        cache_dir=config.bert_cache_dir,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params:,}")

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = DAATrainer(
        model=model,
        config=config,
        event_vocab=event_vocab,
        device=device,
        output_dir=output_dir,
    )
    trainer.train(src_loader, tgt_loader, test_loader, tokenizer)


if __name__ == "__main__":
    main()

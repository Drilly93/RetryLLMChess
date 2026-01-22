# src/train.py
"""
Train script for the Chess Challenge (1M parameter transformer).

Run:
  python -m src.train --output_dir ./my_model --num_train_epochs 3 --per_device_train_batch_size 32

This script:
- loads the dataset dlouapre/lichess_2025-01_1M
- tokenizes games using your custom MyChessTokenizer (tokenizer.py)
- trains your custom ChessForCausalLM (model.py) with Hugging Face Trainer
- saves a submission-ready folder containing:
    config.json, model.safetensors, tokenizer_config.json, vocab.json, model.py, tokenizer.py
"""

import argparse
import os
import shutil

import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, set_seed

from model import ChessConfig, ChessForCausalLM
from tokenizer import MyChessTokenizer


def build_argparser():
    p = argparse.ArgumentParser()

    # Required / common
    p.add_argument("--output_dir", type=str, default="./my_model/final", required=True)
    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)

    # Useful defaults
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    # Data / seq length
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--dataset_split", type=str, default="train")

    # Model hyperparams (keep small for <1M params)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_ctx", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tie_weights", action="store_true", default=True)
    p.add_argument("--max_steps", type=int, default=-1)

    # Performance
    p.add_argument("--fp16", action="store_true", default=False)

    # Paths
    p.add_argument("--vocab_file", type=str, default="vocab.json")

    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = MyChessTokenizer(vocab_file=args.vocab_file)

    config = ChessConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_ctx=args.n_ctx,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = ChessForCausalLM(config)

    dataset = load_dataset("dlouapre/lichess_2025-01_1M", split=args.dataset_split)

    # Tokenize dataset
    # IMPORTANT:
    # - padding to max_length gives attention_mask (1 for real tokens, 0 for PAD)
    # - labels are input_ids, but PAD positions become -100 so they don't contribute to loss
    def tokenize_function(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        # Create labels = input_ids, mask padding labels with -100
        labels = []
        for ids in enc["input_ids"]:
            lab = [(t if t != tokenizer.pad_token_id else -100) for t in ids]
            labels.append(lab)
        enc["labels"] = labels
        return enc

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,  # keep only input_ids/attention_mask/labels
        desc="Tokenizing",
    )

    # Set format for PyTorch
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # TrainingArguments + Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        fp16=args.fp16,
        max_steps=args.max_steps,
        remove_unused_columns=False,  # important with custom model/tokenizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,  # helps saving tokenizer artifacts
    )

    trainer.train()

    # 6) Save final model folder in a submission-ready way
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    # Add auto_map for remote loading (HF Space / leaderboard)
    model.config.auto_map = {
        "AutoConfig": "model.ChessConfig",
        "AutoModelForCausalLM": "model.ChessForCausalLM",
    }
    # For tokenizer: register auto class so tokenizer_config.json has auto_map
    tokenizer.register_for_auto_class("AutoTokenizer")

    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)

    # Copy code files to final dir (required by the challenge)
    shutil.copy("model.py", os.path.join(final_dir, "model.py"))
    shutil.copy("tokenizer.py", os.path.join(final_dir, "tokenizer.py"))

    print(f"\n Training complete. Submission-ready folder: {final_dir}\n")
    print("Contents:", os.listdir(final_dir))


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    main()

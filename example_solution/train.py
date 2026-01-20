"""
Training script for the Chess Challenge.

This script provides a complete training pipeline using the Hugging Face Trainer.
Students can modify this script to experiment with different training strategies.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

# Suppress warnings from third-party libraries (multiprocess has Python 3.14 compat issues)
warnings.filterwarnings("ignore", message="'return' in a 'finally' block")

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

from data import ChessDataCollator, create_train_val_datasets
from model import ChessConfig, ChessForCausalLM
from tokenizer import ChessTokenizer


def count_parameters(model, trainable_only=True):
    """Count the number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a chess-playing language model"
    )
    
    # Model arguments
    parser.add_argument(
        "--n_embd", type=int, default=128,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--n_layer", type=int, default=4,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--n_head", type=int, default=4,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--n_ctx", type=int, default=256,
        help="Maximum context length"
    )
    parser.add_argument(
        "--n_inner", type=int, default=None,
        help="Feed-forward inner dimension (default: 4 * n_embd)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout probability"
    )
    parser.add_argument(
        "--no_tie_weights", action="store_true",
        help="Disable weight tying between embedding and output layers"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_name", type=str, default="dlouapre/lichess_2025-01_1M",
        help="Name of the dataset on Hugging Face Hub"
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Maximum number of training samples"
    )
    parser.add_argument(
        "--val_samples", type=int, default=5000,
        help="Number of validation samples"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory for model and logs"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=32,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=64,
        help="Evaluation batch size per device"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    # Logging arguments
    parser.add_argument(
        "--logging_steps", type=int, default=100,
        help="Logging frequency"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500,
        help="Evaluation frequency"
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000,
        help="Checkpoint saving frequency"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    print("=" * 60)
    print("CHESS CHALLENGE - TRAINING")
    print("=" * 60)
    
    # Build tokenizer from dataset
    print("\nBuilding tokenizer from dataset...")
    tokenizer = ChessTokenizer.build_vocab_from_dataset(
        dataset_name=args.dataset_name,
        min_frequency=500,  # Only keep moves that appear at least 500 times
        max_samples=100000,  # Use 100k games to build vocabulary
    )
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Use the vocab size from tokenizer (override args if provided)
    actual_vocab_size = tokenizer.vocab_size
    
    # Create model configuration
    print("\nCreating model configuration...")
    config = ChessConfig(
        vocab_size=actual_vocab_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_ctx=args.n_ctx,
        n_inner=args.n_inner,
        dropout=args.dropout,
        tie_weights=not args.no_tie_weights,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Print configuration
    print(f"\nModel configuration:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_head: {config.n_head}")
    print(f"  tie_weights: {config.tie_weights}")
    
    # Create model
    print("\nCreating model...")
    model = ChessForCausalLM(config)
    n_params = count_parameters(model)
    print(f"   Total parameters: {n_params:,}")
    
    if n_params > 1_000_000:
        print("WARNING: Model exceeds 1M parameter limit!")
    else:
        print("OK: Model is within 1M parameter limit")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset = create_train_val_datasets(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        max_length=args.n_ctx,
        train_samples=args.max_train_samples,
        val_samples=args.val_samples,
    )
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    
    # Create data collator
    data_collator = ChessDataCollator(tokenizer, max_length=args.n_ctx)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=["none"],
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Copy model.py and tokenizer.py for trust_remote_code loading
    import shutil
    import json
    script_dir = Path(__file__).parent
    shutil.copy(script_dir / "model.py", final_model_dir)
    shutil.copy(script_dir / "tokenizer.py", final_model_dir)
    print("   Copied model.py and tokenizer.py")
    
    # Add auto_map to config.json for AutoModelForCausalLM
    config_path = os.path.join(final_model_dir, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config_dict["auto_map"] = {
        "AutoConfig": "model.ChessConfig",
        "AutoModelForCausalLM": "model.ChessForCausalLM",
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print("   Added auto_map to config.json")
    
    # Add auto_map to tokenizer_config.json for AutoTokenizer
    tokenizer_config_path = os.path.join(final_model_dir, "tokenizer_config.json")
    with open(tokenizer_config_path) as f:
        tokenizer_dict = json.load(f)
    tokenizer_dict["auto_map"] = {
        "AutoTokenizer": ["tokenizer.ChessTokenizer", None],
    }
    with open(tokenizer_config_path, "w") as f:
        json.dump(tokenizer_dict, f, indent=2)
    print("   Added auto_map to tokenizer_config.json")
    
    print("\nTraining complete!")
    print(f"   Model saved to: {final_model_dir}")
    print("   Ready for submission with: python submit.py --model_path " + final_model_dir)


if __name__ == "__main__":
    main()

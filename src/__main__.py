"""
CLI entry point for running evaluation as a module.

Usage:
    python -m src --model ./my_model/final
    python -m src --model username/model-name
"""

import argparse
import sys

from .evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a chess model",
        prog="python -m src",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    result = evaluate_model(args.model, verbose=not args.quiet)
    print()
    print(result.summary())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

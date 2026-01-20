#!/usr/bin/env python3
"""
Submission script for the Chess Challenge.

This script validates and uploads your trained model to the Hugging Face Hub
under the LLM-course organization.

Your model directory must contain:
- config.json: Model configuration with auto_map for custom architecture
- model.safetensors (or pytorch_model.bin): Model weights
- tokenizer_config.json: Tokenizer configuration with auto_map
- vocab.json: Vocabulary file
- model.py: Your custom model architecture (for trust_remote_code)
- tokenizer.py: Your custom tokenizer (for trust_remote_code)

Usage:
    python submit.py --model_path ./my_model --model_name my-chess-model
"""

import argparse
import os
import sys
from pathlib import Path


# Required files for a valid submission
REQUIRED_FILES = {
    "config.json": "Model configuration (must include auto_map)",
    "tokenizer_config.json": "Tokenizer configuration (must include auto_map)",
    "vocab.json": "Vocabulary file",
    "model.py": "Custom model architecture (for trust_remote_code=True)",
    "tokenizer.py": "Custom tokenizer class (for trust_remote_code=True)",
}

# At least one of these weight files must exist
WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin"]


def validate_model_directory(model_path: Path) -> tuple[bool, list[str]]:
    """
    Validate that the model directory contains all required files.
    
    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors = []
    
    # Check required files
    for filename, description in REQUIRED_FILES.items():
        if not (model_path / filename).exists():
            errors.append(f"Missing {filename}: {description}")
    
    # Check weight files (need at least one)
    has_weights = any((model_path / f).exists() for f in WEIGHT_FILES)
    if not has_weights:
        errors.append(f"Missing model weights: need {' or '.join(WEIGHT_FILES)}")
    
    return len(errors) == 0, errors


def validate_auto_map(model_path: Path) -> tuple[bool, list[str]]:
    """
    Validate that config.json and tokenizer_config.json have auto_map fields.
    
    Returns:
        Tuple of (is_valid, list of error messages).
    """
    import json
    
    errors = []
    
    # Check config.json for auto_map
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if "auto_map" not in config:
            errors.append(
                "config.json missing 'auto_map' field. Add:\n"
                '  "auto_map": {\n'
                '    "AutoConfig": "model.YourConfig",\n'
                '    "AutoModelForCausalLM": "model.YourModel"\n'
                '  }'
            )
    
    # Check tokenizer_config.json for auto_map
    tokenizer_config_path = model_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path) as f:
            tokenizer_config = json.load(f)
        if "auto_map" not in tokenizer_config:
            errors.append(
                "tokenizer_config.json missing 'auto_map' field. Add:\n"
                '  "auto_map": {\n'
                '    "AutoTokenizer": ["tokenizer.YourTokenizer", null]\n'
                '  }\n'
                'Note: AutoTokenizer value must be a list [slow_class, fast_class].'
            )
        elif "AutoTokenizer" in tokenizer_config.get("auto_map", {}):
            auto_tok = tokenizer_config["auto_map"]["AutoTokenizer"]
            if isinstance(auto_tok, str):
                errors.append(
                    "tokenizer_config.json auto_map.AutoTokenizer must be a list, not a string.\n"
                    'Change from: "AutoTokenizer": "tokenizer.YourTokenizer"\n'
                    'To: "AutoTokenizer": ["tokenizer.YourTokenizer", null]'
                )
    
    return len(errors) == 0, errors


def count_parameters(model_path: Path) -> int:
    """Count parameters in the model."""
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(
        description="Submit your chess model to the Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Required files in your model directory:
  - config.json          Model configuration with auto_map
  - model.safetensors    Model weights (or pytorch_model.bin)
  - tokenizer_config.json Tokenizer configuration with auto_map
  - vocab.json           Vocabulary file
  - model.py             Custom model architecture
  - tokenizer.py         Custom tokenizer class

Example:
  python submit.py --model_path ./my_model --model_name my-chess-model
        """
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to your trained model directory"
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Name for your model on the Hub (e.g., 'my-chess-model')"
    )
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    organization = "LLM-course"
    
    print("=" * 60)
    print("CHESS CHALLENGE - MODEL SUBMISSION")
    print("=" * 60)
    
    # Check model path exists
    if not model_path.exists():
        print(f"\nError: Model path '{model_path}' does not exist.")
        return 1
    
    # Validate required files
    print("\n[1/5] Checking required files...")
    is_valid, errors = validate_model_directory(model_path)
    if not is_valid:
        print("\nError: Model directory is incomplete:")
        for error in errors:
            print(f"  - {error}")
        print("\nSee example_solution/ for a complete example.")
        return 1
    print("  All required files present.")
    
    # Validate auto_map fields
    print("\n[2/5] Validating auto_map configuration...")
    is_valid, errors = validate_auto_map(model_path)
    if not is_valid:
        print("\nError: Configuration files need auto_map:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("  auto_map configuration valid.")
    
    # Count parameters
    print("\n[3/5] Counting parameters...")
    try:
        n_params = count_parameters(model_path)
        print(f"  Parameters: {n_params:,}")
        if n_params > 1_000_000:
            print(f"\n  WARNING: Model exceeds 1M parameter limit!")
            print(f"  Your model has {n_params:,} parameters.")
            print(f"  It will fail the evaluation parameter check.")
    except Exception as e:
        print(f"\nError: Could not load model to count parameters: {e}")
        return 1
    
    # Hugging Face login
    print("\n[4/5] Checking Hugging Face authentication...")
    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        print("\nError: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        return 1
    
    try:
        user_info = whoami()
        username = user_info["name"]
        print(f"  Logged in as: {username}")
    except Exception:
        print("\n  Not logged in. Starting login process...")
        print("  You need a Hugging Face account and access token.")
        print("  Get your token at: https://huggingface.co/settings/tokens")
        print()
        
        # Interactive login
        from huggingface_hub import login
        try:
            login()
            user_info = whoami()
            username = user_info["name"]
            print(f"\n  Successfully logged in as: {username}")
        except Exception as e:
            print(f"\nError: Login failed: {e}")
            return 1
    
    # Upload to Hub
    print("\n[5/5] Uploading to Hugging Face Hub...")
    repo_id = f"{organization}/{args.model_name}"
    print(f"  Repository: {repo_id}")
    
    api = HfApi()
    
    try:
        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo_id, exist_ok=True)
        
        # Create a model card
        model_card = f"""---
library_name: transformers
tags:
- chess
- llm-course
- chess-challenge
license: mit
---

# {args.model_name}

Chess model submitted to the LLM Course Chess Challenge.

## Submission Info

- **Submitted by**: [{username}](https://huggingface.co/{username})
- **Parameters**: {n_params:,}
- **Organization**: {organization}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True)
```

## Evaluation

This model is evaluated at the [Chess Challenge Arena](https://huggingface.co/spaces/LLM-course/Chess1MChallenge).
"""
        
        # Write model card
        readme_path = model_path / "README.md"
        readme_path.write_text(model_card)
        
        # Upload all files
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            commit_message=f"Chess Challenge submission by {username}",
        )
        
    except Exception as e:
        print(f"\nError: Upload failed: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("SUBMISSION COMPLETE!")
    print("=" * 60)
    print(f"\nYour model is available at:")
    print(f"  https://huggingface.co/{repo_id}")
    print(f"\nSubmitted by: {username}")
    print(f"Parameters: {n_params:,}")
    print(f"\nNext step: Go to the Chess Challenge Arena to run evaluation:")
    print(f"  https://huggingface.co/spaces/LLM-course/Chess1MChallenge")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

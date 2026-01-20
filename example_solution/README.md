# Example Solution

This folder contains a complete reference implementation for the Chess Challenge.

**Use this to understand the expected format** - see how model.py, tokenizer.py, and configuration files should be structured.

## Files Included

| File | Description |
|------|-------------|
| `model.py` | Custom transformer architecture |
| `tokenizer.py` | Custom move-level tokenizer |
| `train.py` | Training script |
| `data.py` | Dataset utilities |
| `config.json` | Model configuration with auto_map |
| `model.safetensors` | Trained model weights |
| `vocab.json` | Tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer configuration with auto_map |
| `special_tokens_map.json` | Special token mappings |

## Model Architecture

This example uses a small GPT-style transformer:

| Parameter | Value |
|-----------|-------|
| Embedding dim | 128 |
| Layers | 4 |
| Attention heads | 4 |
| Context length | 256 |
| Total parameters | ~910K |

## Training Details

The model was trained on the Lichess dataset with:
- 3 epochs
- Batch size 32
- Learning rate 5e-4
- Weight tying (embedding = output layer)

## How to Use This Example

### Load the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./example_solution", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./example_solution", trust_remote_code=True)
```

### Generate a move:

```python
import torch

# Game history in the format: WPe2e4 BPe7e5 WNg1f3 ...
history = "[BOS] WPe2e4 BPe7e5"

inputs = tokenizer(history, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    next_token = outputs.logits[0, -1].argmax()
    
predicted_move = tokenizer.decode([next_token])
print(f"Predicted move: {predicted_move}")
```

## Evaluation

To evaluate this example:

```bash
python -m src.evaluate --model_path ./example_solution
```

## Key Implementation Details

### auto_map Configuration

The `config.json` contains:
```json
{
  "auto_map": {
    "AutoConfig": "model.ChessConfig",
    "AutoModelForCausalLM": "model.ChessForCausalLM"
  }
}
```

The `tokenizer_config.json` contains:
```json
{
  "auto_map": {
    "AutoTokenizer": ["tokenizer.ChessTokenizer", null]
  }
}
```

Note: `AutoTokenizer` requires a list `[slow_class, fast_class]`, not a string!

## Your Turn!

Use this as inspiration, but create your own solution! Ideas to explore:

1. **Architecture changes**: Different number of layers, heads, or embedding dimensions
2. **Training strategies**: Different learning rates, warmup schedules, or optimizers
3. **Data augmentation**: Flip board colors, use different game phases
4. **Tokenization**: Different move representation formats

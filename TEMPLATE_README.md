# Chess Challenge

Train a transformer with less than 1M parameters to play legal chess moves!

## Objective

Design and train a transformer-based language model to predict chess moves. Your model must:

1. **Stay under 1M parameters** - This is the hard constraint!
2. **Create a custom tokenizer** - Design your own move-level tokenizer
3. **Create a custom model architecture** - Build your own transformer
4. **Play legal chess** - The model should learn to generate valid moves
5. **Do NOT use python-chess to filter moves** - The model must generate legal moves on its own

## Dataset

We use the Lichess dataset: [`dlouapre/lichess_2025-01_1M`](https://huggingface.co/datasets/dlouapre/lichess_2025-01_1M)

The dataset uses an extended UCI notation:
- `W`/`B` prefix for White/Black
- Piece letter: `P`=Pawn, `N`=Knight, `B`=Bishop, `R`=Rook, `Q`=Queen, `K`=King  
- Source and destination squares (e.g., `e2e4`)
- Special suffixes: `(x)`=capture, `(+)`=check, `(+*)`=checkmate, `(o)`/`(O)`=castling

Example game:
```
WPe2e4 BPe7e5 WNg1f3 BNb8c6 WBf1b5 BPa7a6 WBb5c6(x) BPd7c6(x) ...
```

---

## Building Your Solution

You need to create **from scratch**:

1. A custom tokenizer class
2. A custom model architecture  
3. A training script
4. Save everything in the correct format

A complete working example is available in `example_solution/` - use it as reference, but build your own!

---

## Step 1: Create a Custom Tokenizer

Your tokenizer must inherit from `PreTrainedTokenizer` and implement the required methods.

### Required Files

Create a file called `tokenizer.py` with your tokenizer class:

```python
import json
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer


class MyChessTokenizer(PreTrainedTokenizer):
    """Custom tokenizer for chess moves."""
    
    # Tell HuggingFace which files to save/load
    vocab_files_names = {"vocab_file": "vocab.json"}
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        **kwargs,
    ):
        # Define special tokens
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"
        
        # Load or create vocabulary
        if vocab_file is not None:
            with open(vocab_file, "r") as f:
                self._vocab = json.load(f)
        else:
            # Create default vocab with special tokens
            self._vocab = {
                "[PAD]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "[UNK]": 3,
            }
        
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
        
        # Call parent init AFTER setting up vocab
        super().__init__(
            pad_token=self.pad_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            unk_token=self.unk_token,
            **kwargs,
        )
    
    @property
    def vocab_size(self) -> int:
        return len(self._vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        return self._vocab.copy()
    
    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens (moves are space-separated)."""
        return text.strip().split()
    
    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get(self.unk_token, 0))
    
    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.unk_token)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        """Save vocabulary to a JSON file."""
        import os
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        with open(vocab_file, "w") as f:
            json.dump(self._vocab, f, indent=2)
        return (vocab_file,)
```

### Building the Vocabulary

You need to build a vocabulary.

It could be written from scratch, or inferred from the dataset:

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("dlouapre/lichess_2025-01_1M", split="train")

# Collect all unique moves
vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
for game in dataset:
    moves = game["text"].split()
    for move in moves:
        if move not in vocab:
            vocab[move] = len(vocab)

print(f"Vocabulary size: {len(vocab)}")

# Save vocabulary
import json
with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)
```

---

## Step 2: Create a Custom Model

Your model must inherit from `PreTrainedModel` and use a config that inherits from `PretrainedConfig`.

### Required Files

Create a file called `model.py` with your model class:

```python
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class MyChessConfig(PretrainedConfig):
    """Configuration for the chess model."""
    
    model_type = "my_chess_model"
    
    def __init__(
        self,
        vocab_size: int = 1500,
        n_embd: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        n_ctx: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.dropout = dropout


class MyChessModel(PreTrainedModel):
    """A simple transformer for chess move prediction."""
    
    config_class = MyChessConfig
    
    def __init__(self, config: MyChessConfig):
        super().__init__(config)
        
        # Token and position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.n_ctx, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=config.n_embd * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layer)
        
        # Output head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (saves parameters!)
        self.lm_head.weight = self.token_emb.weight
        
        self.post_init()
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)
        
        # Causal mask for autoregressive generation
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        
        # Transformer
        x = self.transformer(x, mask=causal_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return CausalLMOutputWithPast(loss=loss, logits=logits)
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
```

### Parameter Budget Tips

With 1M parameters, you need to be careful:

| Component | Formula | Example (128 dim, 1500 vocab) |
|-----------|---------|------------------------------|
| Token embeddings | vocab_size x n_embd | 1500 x 128 = 192,000 |
| Position embeddings | n_ctx x n_embd | 256 x 128 = 32,768 |
| Transformer layer | ~4 x n_embd^2 | ~65,536 per layer |
| LM head | 0 (with weight tying) | 0 |

**Key savings:**
- **Weight tying**: Share token embeddings with output layer (saves vocab_size x n_embd)
- **Smaller vocabulary**: Only include moves that appear in training data
- **Fewer layers**: 4-6 layers is often enough

---

## Step 3: Train Your Model

Create a training script:

```python
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

from model import MyChessConfig, MyChessModel
from tokenizer import MyChessTokenizer

# Load tokenizer with your vocabulary
tokenizer = MyChessTokenizer(vocab_file="vocab.json")

# Create model
config = MyChessConfig(
    vocab_size=tokenizer.vocab_size,
    n_embd=128,
    n_layer=4,
    n_head=4,
)
model = MyChessModel(config)

# Check parameter count
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")
assert n_params < 1_000_000, f"Model too large: {n_params:,} > 1M"

# Load and tokenize dataset
dataset = load_dataset("dlouapre/lichess_2025-01_1M", split="train")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training
training_args = TrainingArguments(
    output_dir="./my_model",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=5e-4,
    save_steps=1000,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Save final model
model.save_pretrained("./my_model/final")
tokenizer.save_pretrained("./my_model/final")
```

---

## Step 4: Prepare for Submission

Your model directory must contain these files:

```
my_model/
  config.json           # Model configuration
  model.safetensors     # Model weights
  tokenizer_config.json # Tokenizer configuration
  vocab.json            # Vocabulary
  model.py              # Your model class
  tokenizer.py          # Your tokenizer class
```

### Adding auto_map for Remote Loading

The `auto_map` field tells HuggingFace how to load your custom classes with `trust_remote_code=True`.

**In config.json**, add:
```json
{
  "auto_map": {
    "AutoConfig": "model.MyChessConfig",
    "AutoModelForCausalLM": "model.MyChessModel"
  },
  ...
}
```

**In tokenizer_config.json**, add:
```json
{
  "auto_map": {
    "AutoTokenizer": "tokenizer.MyChessTokenizer"
  },
  ...
}
```

You can do this programmatically:

```python
# Register for auto loading
model.config.auto_map = {
    "AutoConfig": "model.MyChessConfig",
    "AutoModelForCausalLM": "model.MyChessModel",
}
tokenizer.register_for_auto_class("AutoTokenizer")

# Save
model.save_pretrained("./my_model/final")
tokenizer.save_pretrained("./my_model/final")

# Copy your Python files
import shutil
shutil.copy("model.py", "./my_model/final/model.py")
shutil.copy("tokenizer.py", "./my_model/final/tokenizer.py")
```

---

## Local Evaluation (Optional but Recommended)

Before submitting, you can evaluate your model locally to check its performance. Since the evaluation is **fully deterministic** (fixed seed, deterministic opponent engine), you will get the exact same results locally as on the HuggingFace Space after submission.

```bash
python -m src --model ./my_model/final
```

This runs the same evaluation procedure as the online leaderboard:
- 500 moves against the deterministic opponent
- Same random seed (42)
- Same move generation parameters

Use this to iterate quickly on your model before pushing to HuggingFace!

---

## Step 5: Submit

```bash
python submit.py --model_path ./my_model/final --model_name my-chess-model
```

The script will:
1. Validate all required files are present
2. Check that auto_map is configured
3. Count parameters and warn if over 1M
4. Log you into HuggingFace (if needed)
5. Upload to the LLM-course organization

---

## Evaluation

After submission, go to the [Chess Challenge Arena](https://huggingface.co/spaces/LLM-course/Chess1MChallenge) to run evaluation.

### Evaluation Procedure

1. **Parameter Check**: Model must have < 1M parameters
2. **Security Check**: Code is scanned for illegal python-chess usage  
3. **Game Play**: 500 moves against a deterministic opponent engine
4. **Move Generation**: 3 retries allowed per move (greedy on 1st try, then sampling)
5. **Scoring**: Legal move rate (first try and with retries)

### Scoring

| Metric | Description |
|--------|-------------|
| **Legal Rate (1st try)** | % of moves legal on first attempt |
| **Legal Rate (with retries)** | % of moves legal within 3 attempts |

**Target**: >90% legal rate = excellent performance

---

## Example Solution

A complete working example is in `example_solution/`:

- `model.py` - Full transformer implementation
- `tokenizer.py` - Complete tokenizer class
- `train.py` - Training script with data loading
- `data.py` - Dataset utilities

Use it as reference to understand the expected format and structure.

---

## Rules

1. **< 1M parameters** - Hard limit, checked automatically
2. **No python-chess for move filtering** - Model must generate legal moves on its own
3. **Custom architecture required** - Must include model.py and tokenizer.py
4. **Use the submission script** - Required for leaderboard tracking

Good luck!

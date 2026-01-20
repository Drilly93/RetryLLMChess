# Chess Challenge: 1M Parameter Transformer

Train a transformer (from scratch!) with less than 1M parameters to play legal chess moves.

---

## 1. Overview & Objective

Your model must:

- **Stay under 1M parameters** (hard limit)
- **Create a custom tokenizer** 
- **Create a custom model architecture** (your own transformer)
- **Play legal chess** (model must learn the rules)
- **Do NOT use python-chess to filter moves** (the model must generate legal moves itself)

---

## 2. Dataset & Notation

We use the Lichess dataset: [`dlouapre/lichess_2025-01_1M`](https://huggingface.co/datasets/dlouapre/lichess_2025-01_1M)

**Notation:**
- `W`/`B` prefix for White/Black
- Piece letter: `P`=Pawn, `N`=Knight, `B`=Bishop, `R`=Rook, `Q`=Queen, `K`=King
- Source and destination squares (e.g., `e2e4`)
- Special suffixes: `(x)`=capture, `(+)`=check, `(+*)`=checkmate, `(o)`/`(O)`=castling

Example game:
```
WPe2e4 BPe7e5 WNg1f3 BNb8c6 WBf1b5 BPa7a6 WBb5c6(x) BPd7c6(x) ...
```

---

## 3. Directory Structure

Your project should look like this:

```
my_model/
  config.json
  model.safetensors
  tokenizer_config.json
  vocab.json
  model.py
  tokenizer.py
```

---

## 4. Step-by-Step Instructions

### Step 1: Build Your Tokenizer

Create `tokenizer.py` implementing a subclass of `PreTrainedTokenizer`, say MyChessTokenizer.

**Build the vocabulary:**

One possibility is to look at the dataset, but it's by far not the only option:

```python
from datasets import load_dataset
import json

dataset = load_dataset("dlouapre/lichess_2025-01_1M", split="train")
vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
for game in dataset:
    for move in game["text"].split():
        if move not in vocab:
            vocab[move] = len(vocab)
with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)
```

### Step 2: Build Your Model

Create `model.py` implementing a subclass of `PreTrainedModel`, say MyChessModel, and a config class, say MyChessConfig.

**Tips:**
- Use weight tying to save parameters.
- Keep the vocabulary small.
- 4-6 transformer layers is usually enough.

### Step 3: Training

Create `train.py` to train your model:
```python
from model import MyChessConfig, MyChessModel
from tokenizer import MyChessTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

tokenizer = MyChessTokenizer(vocab_file="vocab.json")
config = MyChessConfig(vocab_size=tokenizer.vocab_size, n_embd=128, n_layer=4, n_head=4)
model = MyChessModel(config)

dataset = load_dataset("dlouapre/lichess_2025-01_1M", split="train")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./my_model",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=5e-4,
    save_steps=1000,
    logging_steps=100,
)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()
model.save_pretrained("./my_model/final")
tokenizer.save_pretrained("./my_model/final")
```

### Step 4: Prepare for Submission

Your model directory (`my_model/final/`) **must** contain:
```
config.json           # Model configuration
model.safetensors     # Model weights
tokenizer_config.json # Tokenizer configuration
vocab.json            # Vocabulary
model.py              # Your model class
tokenizer.py          # Your tokenizer class
```

#### Add `auto_map` for remote loading
Edit `config.json`:
```json
  "auto_map": {
    "AutoConfig": "model.MyChessConfig",
    "AutoModelForCausalLM": "model.MyChessModel"
  }
```
Edit `tokenizer_config.json`:
```json
  "auto_map": {
    "AutoTokenizer": "tokenizer.MyChessTokenizer"
  }
```
Or do it programmatically:
```python
model.config.auto_map = {
    "AutoConfig": "model.MyChessConfig",
    "AutoModelForCausalLM": "model.MyChessModel",
}
tokenizer.register_for_auto_class("AutoTokenizer")
model.save_pretrained("./my_model/final")
tokenizer.save_pretrained("./my_model/final")
import shutil
shutil.copy("model.py", "./my_model/final/model.py")
shutil.copy("tokenizer.py", "./my_model/final/tokenizer.py")
```

---

### Step 5: Local Evaluation (Recommended)

Before submitting, evaluate your model locally:
```bash
python -m src --model ./my_model/final
```
This runs the same evaluation as the leaderboard (500 moves, deterministic seed).

---

### Step 6: Submit

Submit your model to the leaderboard:
```bash
python submit.py --model_path ./my_model/final --model_name your-model-name
```
The script will:
- Validate all required files
- Check auto_map
- Count parameters
- Log you into HuggingFace (if needed)
- Upload to the LLM-course organization

---

## 5. Evaluation & Leaderboard

After submission, go to the [Chess Challenge Arena](https://huggingface.co/spaces/LLM-course/Chess1MChallenge) to run evaluation.

**Evaluation steps:**
1. Parameter check (<1M)
2. Security check (no python-chess for move filtering)
3. 500 moves against a deterministic opponent
4. 3 retries per move (greedy, then sampling)
5. Scoring: legal move rate (first try and with retries)

**Scoring Table:**
| Metric | Description |
|--------|-------------|
| **Legal Rate (1st try)** | % of moves legal on first attempt |
| **Legal Rate (with retries)** | % of moves legal within 3 attempts |

**Target:** >90% legal rate = excellent

---

## 6. Example Solution

See `example_solution/` for a full working reference:
- `model.py`, `tokenizer.py`, `train.py`, `data.py`

---

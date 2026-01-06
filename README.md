---
title: Chess Challenge Arena
emoji: ♟️
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
---

# Chess Challenge Arena

This Space hosts the evaluation arena for the LLM Chess Challenge.

## Features

- **Interactive Demo**: Test any submitted model against Stockfish
- **Leaderboard**: See rankings of all submitted models
- **Statistics**: View detailed performance metrics

## How to Submit

Students should push their trained models to this organization:

```python
from chess_challenge import ChessForCausalLM, ChessTokenizer

model.push_to_hub("your-model-name", organization="LLM-course")
tokenizer.push_to_hub("your-model-name", organization="LLM-course")
```

Models will be automatically evaluated and added to the leaderboard.

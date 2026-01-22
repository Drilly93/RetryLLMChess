# test_smoke.py
# Quick smoke test for your custom tokenizer + model.
# What it checks:
# 1) Tokenizer produces tokens and can decode MOVE_END as whitespace.
# 2) Model forward pass works (loss + logits shapes).
# 3) A tiny "training" loop (a few gradient steps) runs and the loss should
#    generally go down a bit (not guaranteed every step, but trend should improve).

import torch
from model import ChessConfig, ChessForCausalLM
from tokenizer import MyChessTokenizer


def main():
    torch.manual_seed(0)

    # Load tokenizer
    tokenizer = MyChessTokenizer("vocab.json")

    # --- Tokenizer sanity checks ---
    toks = tokenizer.tokenize(tokenizer.bos_token + " WPe2e4")
    print("Tokens for '[BOS] WPe2e4':", toks)

    move_end_id = tokenizer.convert_tokens_to_ids("MOVE_END")
    print("decode(MOVE_END) repr:", repr(tokenizer.decode([move_end_id])))

    # -------------------------------
    # 🔍 SHOW TOKENIZATION DETAILS
    # -------------------------------
    texts = [
        "WPe2e4 BPe7e5 WNg1f3",
        "WPe2e4 WBf1b5(x) BPa7a6",
    ]

    print("\n=== TOKENIZER OUTPUT ===")
    for i, t in enumerate(texts):
        tokens = tokenizer.tokenize(t)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        decoded = tokenizer.decode(ids)

        print(f"\nExample {i}")
        print("raw text   :", t)
        print("tokens     :", tokens)
        print("token ids  :", ids)
        print("decoded    :", repr(decoded))

    # Build tiny model (very small for fast test)
    config = ChessConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_ctx=32,
        dropout=0.0,
    )
    model = ChessForCausalLM(config)
    model.train()

    enc = tokenizer(texts, padding="max_length", max_length=32, return_tensors="pt")

    labels = enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # --- Forward pass check ---
    out = model(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        labels=labels,
    )
    print("\n✅ forward OK")
    print("loss:", float(out.loss))
    print("logits shape:", tuple(out.logits.shape))

    # --- Tiny training loop ---
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\nTiny training loop (10 steps):")
    for step in range(10):
        opt.zero_grad(set_to_none=True)
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
        )
        loss = out.loss
        loss.backward()
        opt.step()
        print(f"  step {step:02d} | loss {float(loss):.4f}")

    # --- Next-token sanity check ---
    model.eval()
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        next_ids = out.logits[:, -1, :].argmax(dim=-1)
        print("\nGreedy next token ids:", next_ids.tolist())
        print(
            "Greedy next token strings:",
            [tokenizer.decode([i]) for i in next_ids.tolist()],
        )


if __name__ == "__main__":
    main()

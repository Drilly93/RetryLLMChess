import json

vocab = {}

def add(tok):
    if tok not in vocab:
        vocab[tok] = len(vocab)

# Special tokens (order matters)
add("[PAD]")
add("[BOS]")
add("[EOS]")
add("[UNK]")

# Colors
add("W")
add("B")

# Pieces
add("P")
add("N")
add("BISHOP")
add("R")
add("Q")
add("K")

# Squares
files = "abcdefgh"
ranks = "12345678"
for f in files:
    for r in ranks:
        add(f + r)

# Symbols
add("x")
add("+")
add("#")
add("O-O")
add("O-O-O")
add("MOVE_END")

# Promotions
add("prom_Q")
add("prom_R")
add("prom_B")
add("prom_N")

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)

print("✅ vocab.json created")
print("Vocab size:", len(vocab))

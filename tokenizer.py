import json
import re
from transformers import PreTrainedTokenizer


class MyChessTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}

    def __init__(self, vocab_file, **kwargs):
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id_to_token = {i: t for t, i in self.vocab.items()}

        # Special tokens expected to exist in vocab.json
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("bos_token", "[BOS]")
        kwargs.setdefault("eos_token", "[EOS]")
        kwargs.setdefault("unk_token", "[UNK]")
        super().__init__(**kwargs)

        # Quick helpers
        self._sq_re = re.compile(r"^[a-h][1-8]$")
        self._piece_map = {
            "P": "P",
            "N": "N",
            "B": "BISHOP",   # use your vocab entry name
            "R": "R",
            "Q": "Q",
            "K": "K",
        }
        self._prom_map = {
            "Q": "prom_Q",
            "R": "prom_R",
            "B": "prom_B",
            "N": "prom_N",
        }

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text):
        chunks = text.strip().split()
        out = []
        for c in chunks:
            # Ignore literal special tokens inserted as text by evaluation
            if c in {self.bos_token, self.eos_token, self.pad_token}:
                continue
            out.extend(self._encode_one_move_to_tokens(c))
        return out

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, idx):
        return self.id_to_token.get(idx, self.unk_token)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def convert_tokens_to_string(self, tokens):
        out = []
        for t in tokens:
            if t == "MOVE_END":
                out.append(" ")
            elif t == self.eos_token:
                out.append(" ")   # ou " "
            elif t in {self.pad_token, self.bos_token}:
                continue
            else:
                out.append(t)
        return "".join(out)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        name = "vocab.json" if filename_prefix is None else filename_prefix + "-vocab.json"
        path = save_directory.rstrip("/") + "/" + name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)
        return (path,)

    def _encode_one_move_to_tokens(self, move_str):
        """
        We do NOT keep each chess move as a single token. Instead, we decompose a move into
        a short sequence of atomic tokens (color, piece, from-square, to-square, and optional
        flags like capture/check/mate/promotion/castling). This keeps the vocabulary small
        and forces the model to learn legal composition rules.

        Examples:
          "WPe2e4"           -> ["W","P","e2","e4","MOVE_END"]
          "WBb5c6(x)"        -> ["W","BISHOP","b5","c6","x","MOVE_END"]
          "WNg1f3(+)"        -> ["W","N","g1","f3","+","MOVE_END"]
          "W(O)" / "W(o)"    -> ["W","O-O-O","MOVE_END"]  or ["W","O-O","MOVE_END"]
        """
        toks = []

        # --- Castling: "(o)" short or "(O)" long ---
        # We assume the move string contains W/B somewhere at the start.
        if "(o)" in move_str or "(O)" in move_str:
            if move_str.startswith("W"):
                toks.append("W")
            elif move_str.startswith("B"):
                toks.append("B")
            else:
                toks.append(self.unk_token)

            if "(o)" in move_str:
                toks.append("O-O")
            else:
                toks.append("O-O-O")

            toks.append("MOVE_END")
            return toks

        # --- Standard UCI-like extended move: "WPe2e4..." ---
        # Minimal parsing:
        #   0: color (W/B)
        #   1: piece (P/N/B/R/Q/K)
        #   2-3: from square
        #   4-5: to square
        if len(move_str) >= 6 and move_str[0] in "WB" and move_str[1] in "PNBRQK":
            color = move_str[0]
            piece_char = move_str[1]
            from_sq = move_str[2:4]
            to_sq = move_str[4:6]
            rest = move_str[6:]

            toks.append(color)
            toks.append(self._piece_map.get(piece_char, self.unk_token))

            toks.append(from_sq if self._sq_re.match(from_sq) else self.unk_token)
            toks.append(to_sq if self._sq_re.match(to_sq) else self.unk_token)

            # capture
            if "(x)" in rest:
                toks.append("x")

            # check / mate
            # dataset examples mention "(+)" and "(+*)" ; some variants might use "(#)"
            if "(+*)" in rest or "(#)" in rest:
                toks.append("#")
            elif "(+)" in rest:
                toks.append("+")

            # promotion: look for "=Q" "=R" "=B" "=N"
            pm = re.search(r"=([QRBN])", rest)
            if pm:
                promo = pm.group(1)
                toks.append(self._prom_map.get(promo, self.unk_token))

            toks.append("MOVE_END")
            return toks

        # --- Fallback ---
        return [self.unk_token, "MOVE_END"]

from typing import Iterable

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        
        if special_tokens:
            for tok in dict.fromkeys(special_tokens):
                tok_bytes = tok.encode("utf-8")
                if tok_bytes not in self.vocab.values():
                    next_id = max(self.vocab.keys(), default=-1) + 1
                    self.vocab[next_id] = tok_bytes
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        vocab = {int(idx): bytes(token, encode="utf-8") for idx, token in vocab_json.items()}

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = []
            for line in f:
                if line.strip() and not line.startswith("#"):
                    pair = tuple(line.strip().split())
                    merges.add((bytes(pair[0], encoding="utf-8"), bytes(pair[1], encoding='utf-8')))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str):
        max_len = max(len(b.decode("utf-8", errors="replace")) for b in self.vocab.values())

        byte_sequence = text.encode("utf-8")

        tokens = []

        i = 0
        while i < len(byte_sequence):
            for j in range(max_len, 0, -1):
                sub_token = byte_sequence[i:i + j]
                if sub_token in self.vocab.values():
                    token_id = list(self.vocab.keys())[list(self.vocab.values()).index(sub_token)]
                    tokens.append(token_id)
                    i = i + j
                    break

            else:
                raise ValueError(f"Cannot encode token starting at position {i}: {byte_sequence[i:i+10]}...")
        return tokens
            

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        byte_chunks = [self.vocab[i] for i in ids]
        return b"".join(byte_chunks).decode("utf-8", errors="replace")
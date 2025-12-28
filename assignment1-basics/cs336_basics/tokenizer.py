import json
from typing import Iterable

import regex as re

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        for tok in dict.fromkeys(self.special_tokens):
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.vocab.values():
                next_id = max(self.vocab.keys(), default=-1) + 1
                self.vocab[next_id] = tok_bytes
    
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.byte_to_id = {b: i for i, b in self.vocab.items()}
        self.bpe_rank = {pair: rank for rank, pair in enumerate(self.merges)}

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
    
    def _split_special(self, text: str) -> list[tuple[bool, str]]:
        if not self.special_tokens:
            return [(False, text)]
        
        specials = sorted(self.special_tokens, key=len, reverse=True) # must be ranked by length
        parts: list[tuple[bool, str]] = []
        i = 0
        start = 0
        while i < len(text):
            matched = None
            for tok in specials:
                if text.startswith(tok, i):
                    matched = tok
                    break
            if matched:
                if i > start:
                    parts.append((False, text[start:i]))
                parts.append((True, matched))
                i += len(matched)
                start = i
            else:
                i += 1
        if start < len(text):
            parts.append((False, text[start:]))
        return parts

    def _get_pairs(self, word: list[bytes]) -> set[tuple[bytes, bytes]]:
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs

    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        word = [bytes([b]) for b in token_bytes]
        if not word:
            return []
        pairs = self._get_pairs(word)
        while pairs:
            ranked = [(self.bpe_rank.get(p), p) for p in pairs if p in self.bpe_rank]
            if not ranked:
                break

            _, best = min(ranked, key=lambda x: x[0])

            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            pairs = self._get_pairs(word)
        return word

    def encode(self, text: str):
        tokens: list[int] = []

        for is_special, segment in self._split_special(text):
            if is_special:
                tok_bytes = segment.encode("utf-8")
                tok_id = self.byte_to_id.get(tok_bytes)
                if tok_id is None:
                    raise ValueError(f"Unkown special token: {segment}")
                tokens.append(tok_id)
                continue
            
            for match in self.pat.finditer(segment):
                piece = match.group(0)
                piece_bytes = piece.encode("utf-8")
                for bpe_token in self._bpe(piece_bytes):
                    tok_id = self.byte_to_id.get(bpe_token)
                    if tok_id is None:
                        raise ValueError(f"Unknown token bytes: {bpe_token!r}")
                    tokens.append(tok_id)
        return tokens
            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        byte_chunks = [self.vocab[i] for i in ids]
        return b"".join(byte_chunks).decode("utf-8", errors="replace")
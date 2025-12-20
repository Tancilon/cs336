import regex as re
import threading
import os
from typing import BinaryIO
from multiprocessing import Pool
from collections import Counter


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def chunk_process(args):
    input_path, start, end, pat, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    if special_tokens:
        split_pat = "|".join(re.escape(tok) for tok in special_tokens)
        parts = re.split(split_pat, chunk)
    else:
        parts = [chunk]

    counts = Counter()
    for part in parts:
        for m in re.finditer(pat, part):
            counts[m.group(0)] += 1
    return counts

def bytes_word(token: str) -> tuple[bytes, ...]:
    b = token.encode("utf-8")
    return tuple(bytes([x]) for x in b)

def merged_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)

def bpe_merges_from_counts(counts: Counter, vocab_size: int, vocab: dict[int, bytes], next_id: int):
    # word_counts: dict{tuple[bytes, ...], int}
    word_counts = {}
    for tok, freq in counts.items():
        word = bytes_word(tok)
        word_counts[word] = word_counts.get(word, 0) + freq
    
    merges = []
    while len(vocab) < vocab_size:
        pair_counts = Counter()
        for word, freq in word_counts.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i+1])] += freq
        
        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(best_pair)

        new_vocab = best_pair[0] + best_pair[1]
        vocab[next_id] = new_vocab
        next_id += 1

        new_word_counts = {}
        for word, freq in word_counts.items():
            merged = merged_word(word, best_pair)
            new_word_counts[merged] = new_word_counts.get(merged, 0) + freq
        word_counts = new_word_counts
    return vocab, merges

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    return:
        - vocab: dict[int, bytes]
        - merges: list[tuple[bytes, bytes]]
    """
    vocab = {i:bytes([i]) for i in range(256)}
    next_id = 256
    special_tokens = list(dict.fromkeys(special_tokens))
    for tok in special_tokens:
        tok_bytes = tok.encode("utf-8")
        vocab[next_id] = tok_bytes
        next_id += 1
    print(f"[ Debug ] Initialize vocab size: {len(vocab)}")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=4, split_special_token=b"<|endoftext|>")

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # pattern = "|".join(regex.escape(special_tok) for special_tok in special_tokens)
    # matches = re.finditer(pattern, content)
    tasks = [(input_path, s, e, PAT, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]
    with Pool(4) as pool:
        counters = pool.map(chunk_process, tasks)
    total_counts = Counter()
    for c in counters:
        total_counts.update(c)
       
    vocab, merges = bpe_merges_from_counts(total_counts, vocab_size, vocab, next_id)

    return vocab, merges

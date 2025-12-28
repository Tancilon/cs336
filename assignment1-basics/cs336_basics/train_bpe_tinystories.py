from train_bpe import train_bpe

if __name__ == "__main__":
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, "<|endoftext|>")
    logest_token = max(vocab.value(), key=len)
    print(f"The longest token has the length of {logest_token}")
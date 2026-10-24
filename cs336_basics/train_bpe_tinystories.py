from pathlib import Path
import time

from cs336_basics.tokenization import tokenization

input_path = Path("./data/TinyStoriesV2-GPT4-train.txt")
max_vocab_size = 10,000
special_tokens : list[str] = ["<|endoftext|>"]

start = time.perf_counter()

vocab,merges = tokenization(input_path,max_vocab_size,special_tokens)

elapsed = time.perf_counter() - start

print("Training time:", elapsed)
print("Vocabulary size:", len(vocab))
print("Number of merges:", len(merges))
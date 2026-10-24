from pathlib import Path
import time
import pickle

from cs336_basics.bpe.tokenization2 import train_bpe

input_path = Path("./data/owt_valid.txt")

max_vocab_size = 32000

special_tokens : list[str] = ["<|endoftext|>"]

start = time.perf_counter()

vocab,merges = train_bpe(input_path,max_vocab_size,special_tokens)

elapsed = time.perf_counter() - start

print("Training time:", elapsed)
print("Vocabulary size:", len(vocab))
print("Number of merges:", len(merges))


# save the vocab and merges
output_dir = Path("./artifacts")
output_dir.mkdir(parents=True, exist_ok=True)

with open (output_dir/"vocab.pkl","wb") as f :
    pickle.dump(vocab,f)
    
with open (output_dir/"merges.pkl","wb") as f :
    pickle.dump(merges,f)


print("Vocabulary and merges saved successfully.")
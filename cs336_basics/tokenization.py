
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

import regex as re
import os
from typing import BinaryIO


from pathlib import Path

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

def  tokenization (
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # pretokenization
    # cut off the special token like  <|endoftext|>
    
    # counts :{"tuple",int}
    counts: dict[tuple[bytes, ...], int] = {}
    with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                print(f"字节范围：[{start}, {end})")
                print("字符数：", len(chunk))
                print("文本开头：", repr(chunk[:80]))
                
                print("start Pre-tokenization:")
                for st in special_tokens :
                    texts = chunk.split(st)
                    
                for text in texts:
                    for match in re.finditer(PAT, text):
                        pretoken = match.group()
                        pretoken_bytes = pretoken.encode("utf-8")
                        key = tuple(bytes([byte]) for byte in pretoken_bytes)
                        counts[key] = counts.get(key, 0) + 1
                print("Finish Pre-tokenization")
    
    
                            
    # Merges
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)  }
    
    for special_token in special_tokens :
        i = len(vocab)
        vocab[i] = special_token.encode("utf-8")

    merges: list[tuple[bytes,bytes]] = []

    while len(vocab) < vocab_size:
        
        pair_counts : dict[tuple[bytes,bytes],int] = {}
        
        # for words,frequency in counts.items():
        #     for i in range(len(words) - 1):
        #         pair = (words[i],words[i + 1])
        #         pair_counts[pair] = pair_counts.get(pair,0) + frequency
        
        # Merge adjacent byte tokens
        for words,frequency in counts.items():
            for pair in zip(words[:-1],words[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + frequency

        if not pair_counts:
            break
        
        # select the most frequent bytes pair if equal choose the 字典序最大
        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], pair),
        )
        
        merged_token = best_pair[0] + best_pair[1]
        
        vocab[len(vocab)] = merged_token
        
        merges.append(best_pair)
        
        new_counts :  dict[tuple[bytes, ...], int] = {}
        
        for word,frequency in counts.items():
            new_word = []
            i = 0
            
            while i < len(word) :
                if word[i: i + 2] == best_pair:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_type = tuple(new_word)
            new_counts[new_type] = new_counts.get(new_type,0) + frequency
            
        counts = new_counts 
        
    return vocab,merges
           
                    
                    
        
        

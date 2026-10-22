import os
from collections import Counter
from itertools import pairwise
from typing import BinaryIO

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    返回文件分块边界，位置以字节为单位。
    边界重合时，实际块数可能少于期望块数。
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [
        i * chunk_size
        for i in range(desired_num_chunks + 1)
    ]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)

            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def tokenization(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 提前准备正则匹配工具
    token_pattern = re.compile(PAT)

    if special_tokens:
        special_pattern = re.compile(
            "|".join(
                re.escape(st)
                for st in sorted(
                    special_tokens,
                    key=len,
                    reverse=True,
                )
            )
        )
    else:
        special_pattern = None

    # 先统计字符串，不对每次出现都进行字节转换
    text_counts: Counter[str] = Counter()

    with open(input_path, "rb") as f:
        num_processes = 4  # 沿用原变量名，这里只决定分块数量

        if len(special_tokens) == 1:
            boundaries = find_chunk_boundaries(
                f,
                num_processes,
                special_tokens[0].encode("utf-8"),
            )
        else:
            # 没有或有多个特殊标记时，先使用一个完整块
            f.seek(0, os.SEEK_END)
            boundaries = [0, f.tell()]

        # 单进程逐块读取
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode(
                "utf-8",
                errors="ignore",
            )

            if special_pattern is not None:
                texts = special_pattern.split(chunk)
            else:
                texts = [chunk]

            for text in texts:
                text_counts.update(
                    match.group()
                    for match in token_pattern.finditer(text)
                )

    # 256 种单字节对象提前创建，之后重复使用
    byte_tokens = tuple(bytes([i]) for i in range(256))

    # 每种不同片段只编码、转换一次
    counts: dict[tuple[bytes, ...], int] = {}

    for text, frequency in text_counts.items():
        key = tuple(
            byte_tokens[byte]
            for byte in text.encode("utf-8")
        )
        counts[key] = frequency

    # 后面只使用 counts，释放字符串频率表
    del text_counts

    # 初始化词表
    vocab: dict[int, bytes] = {
        i: byte_tokens[i]
        for i in range(256)
    }

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        # 统计相邻 token 对
        pair_counts: dict[tuple[bytes, bytes], int] = {}

        for word, frequency in counts.items():
            for pair in pairwise(word):
                pair_counts[pair] = (
                    pair_counts.get(pair, 0) + frequency
                )

        if not pair_counts:
            break

        # 优先比较频率；频率相同，再比较字典序
        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], pair),
        )

        left, right = best_pair
        merged_token = left + right

        vocab[len(vocab)] = merged_token
        merges.append(best_pair)

        new_counts: dict[tuple[bytes, ...], int] = {}

        for word, frequency in counts.items():
            # 缺少任意一侧 token，就不可能发生本轮合并
            if left not in word or right not in word:
                new_key = word
            else:
                new_word = []
                i = 0
                word_length = len(word)

                while i < word_length:
                    if (
                        i + 1 < word_length
                        and word[i] == left
                        and word[i + 1] == right
                    ):
                        new_word.append(merged_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                new_key = tuple(new_word)

            # 完整处理一个片段后，再记录频率
            new_counts[new_key] = (
                new_counts.get(new_key, 0) + frequency
            )

        counts = new_counts

    return vocab, merges
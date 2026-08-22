import multiprocessing as mp
import os
from collections import Counter
from collections.abc import Iterator
from functools import partial
from typing import BinaryIO

import regex as re


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


def pre_tokenize_worker(
    text: str,
    st_pattern: str,
) -> tuple[Counter[tuple[bytes, ...]], Counter[tuple[bytes, bytes]]]:
    pretokens: Counter[tuple[bytes, ...]] = Counter()
    pair_count: Counter[tuple[bytes, bytes]] = Counter()

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in re.split(st_pattern, text):
        for match in re.finditer(PAT, chunk):
            pt = match.group().encode("utf-8")
            pt = tuple(pt[i : i + 1] for i in range(len(pt)))  # tuple[bytes, ...]
            pretokens[pt] += 1
    for pt in pretokens:
        # 统计所有相邻pair的出现频率
        for i in range(len(pt) - 1):
            pair_count[(pt[i], pt[i + 1])] += pretokens[pt]

    return pretokens, pair_count


class BPE:
    def __init__(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ):
        self._input_path = input_path
        self._vocab_size = vocab_size
        self._special_tokens = special_tokens
        self._vocab: dict[int, bytes] = {c: bytes([c]) for c in range(256)}
        for st in special_tokens:
            self._vocab[len(self._vocab)] = st.encode("utf-8")
        self._pretokens: Counter[tuple[bytes, ...]] = Counter()
        self._merges: list[tuple[bytes, bytes]] = []
        self._pair_count: Counter[tuple[bytes, bytes]] = Counter()

    def _read_text(self, num_chunks: int) -> Iterator[str]:
        with open(self._input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                yield chunk

    def _pre_tokenize(self):
        num_processes = max(1, mp.cpu_count() - 2)
        file_size = os.path.getsize(self._input_path)
        max_chunk_size = 32 * 1024 * 1024  # 32MB
        num_chunks = min(4 * num_processes, max(1, (file_size + max_chunk_size - 1) // max_chunk_size))
        st_pattern = "|".join([re.escape(st) for st in self._special_tokens])
        worker = partial(pre_tokenize_worker, st_pattern=st_pattern)
        with mp.Pool(processes=num_processes) as pool:
            for words, pair_count in pool.imap_unordered(worker, self._read_text(num_chunks)):
                self._pretokens += words
                self._pair_count += pair_count

    def _merge_once(self):
        # 把频率最高的pair合并成一个新符号
        pair_most = max(self._pair_count, key=lambda k: (self._pair_count[k], k))
        self._merges.append(pair_most)
        bytes_merged = pair_most[0] + pair_most[1]
        self._vocab[len(self._vocab)] = bytes_merged

        # pretoken内部的合并
        for pt, cnt in self._pretokens.copy().items():
            pto = pt
            n = len(pt)
            i = 0
            while i < n - 1:
                if pt[i : i + 2] == pair_most:
                    self._pair_count[(pt[i], pt[i + 1])] -= cnt
                    if i > 0:
                        self._pair_count[(pt[i - 1], pt[i])] -= cnt
                        self._pair_count[(pt[i - 1], bytes_merged)] += cnt
                    if i + 2 < n:
                        self._pair_count[(pt[i + 1], pt[i + 2])] -= cnt
                        self._pair_count[(bytes_merged, pt[i + 2])] += cnt
                    # 把所有该pair的出现都替换成新符号
                    pt = pt[:i] + (bytes_merged,) + pt[i + 2 :]
                    n = len(pt)
                i += 1
            if n != len(pto):
                self._pretokens[pt] = self._pretokens.pop(pto)

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self._pre_tokenize()
        while len(self._vocab) < self._vocab_size:
            self._merge_once()
        return self._vocab, self._merges

import multiprocessing as mp
import pickle
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import inf

import regex as re


def bpe(
    pt: tuple[bytes, ...],
    merge_to_rank: dict[tuple[bytes, bytes], int],
) -> tuple[bytes, ...]:
    # pretoken内的合并
    while True:
        if len(pt) <= 1:
            break
        rank_idx_list = [(merge_to_rank.get((pt[i], pt[i + 1]), inf), i) for i in range(len(pt) - 1)]
        best_rank, idx = min(rank_idx_list)
        if best_rank == inf:
            break
        pt = pt[:idx] + (pt[idx] + pt[idx + 1],) + pt[idx + 2 :]
    return pt


def encode(
    text: str,
    vocab_inv: dict[bytes, int],
    merge_to_rank: dict[tuple[bytes, bytes], int],
    st_set: set[str],
    st_pattern: str,
) -> list[int]:
    ids: list[int] = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    chunks = re.split(st_pattern, text) if st_set else [text]
    for chunk in chunks:
        if chunk in st_set:
            ids.append(vocab_inv[chunk.encode("utf-8")])
        else:
            for match in re.finditer(PAT, chunk):
                pt = match.group().encode("utf-8")
                pt = tuple(pt[i : i + 1] for i in range(len(pt)))  # tuple[bytes, ...]
                pt = bpe(pt, merge_to_rank)
                ids += [vocab_inv[bs] for bs in pt]
    return ids


@dataclass(frozen=True)
class WorkerState:
    vocab_inv: dict[bytes, int]
    merge_to_idx: dict[tuple[bytes, bytes], int]
    st_set: set[str]
    st_pattern: str


_state: WorkerState


def init_worker(state: WorkerState):
    global _state
    _state = state


def encode_worker(text: str) -> list[int]:
    return encode(text, _state.vocab_inv, _state.merge_to_idx, _state.st_set, _state.st_pattern)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self._vocab = vocab
        self._vocab_inv: dict[bytes, int] = {bs: id for id, bs in vocab.items()}
        self._merge_to_rank: dict[tuple[bytes, bytes], int] = {m: i for i, m in enumerate(merges)}
        self._st_set: set[str] = set(special_tokens or [])
        self._st_pattern = f"""({"|".join([re.escape(st) for st in sorted(self._st_set, key=len, reverse=True)])})"""

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        return encode(text, self._vocab_inv, self._merge_to_rank, self._st_set, self._st_pattern)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        num_processes = max(1, mp.cpu_count() - 2)
        with mp.Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(WorkerState(self._vocab_inv, self._merge_to_rank, self._st_set, self._st_pattern),),
        ) as pool:
            for ids in pool.imap(encode_worker, iterable, chunksize=2):
                yield from ids

    def decode(self, ids: list[int]) -> str:
        return b"".join([self._vocab[id] for id in ids]).decode("utf-8", errors="replace")

import pickle
from collections.abc import Iterable, Iterator
from math import inf

import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.vocab_inv: dict[bytes, int] = {bs: id for id, bs in vocab.items()}
        self.merges = merges
        self.merge_to_rank: dict[tuple[bytes, bytes], int] = {m: i for i, m in enumerate(merges)}
        self.special_tokens: set[str] = set(special_tokens or [])

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

    def _bpe(self, pt: tuple[bytes, ...]) -> tuple[bytes, ...]:
        # pretoken内的合并
        while True:
            if len(pt) <= 1:
                break
            rank_idx_list = [(self.merge_to_rank.get((pt[i], pt[i + 1]), inf), i) for i in range(len(pt) - 1)]
            best_rank, idx = min(rank_idx_list)
            if best_rank == inf:
                break
            pt = pt[:idx] + (pt[idx] + pt[idx + 1],) + pt[idx + 2 :]
        return pt

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            st_pattern = "|".join([re.escape(st) for st in sorted(self.special_tokens, key=len, reverse=True)])
            st_pattern = f"({st_pattern})"
            chunks = re.split(st_pattern, text)
        else:
            chunks = [text]

        ids: list[int] = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in chunks:
            if chunk in self.special_tokens:
                ids.append(self.vocab_inv[chunk.encode("utf-8")])
            else:
                for match in re.finditer(PAT, chunk):
                    pt = match.group().encode("utf-8")
                    pt = tuple(pt[i : i + 1] for i in range(len(pt)))  # tuple[bytes, ...]
                    pt = self._bpe(pt)
                    ids += [self.vocab_inv[bs] for bs in pt]
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")

import os
from collections import Counter

import regex as re


class BPE:
    def __init__(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ):
        self._input_path = input_path
        self._vocab_size = vocab_size
        self._vocab: dict[int, bytes] = {c: bytes([c]) for c in range(256)}
        for st in special_tokens:
            self._vocab[len(self._vocab)] = st.encode("utf-8")
        self._pretokens: Counter[tuple[bytes, ...]] = Counter()
        self._merges: list[tuple[bytes, bytes]] = []

    def _pre_tokenize(self):
        with open(self._input_path, encoding="utf-8") as f:
            text = f.read()
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for match in re.finditer(PAT, text):
            pt = match.group().encode("utf-8")
            pt = tuple(pt[i : i + 1] for i in range(len(pt)))  # tuple[bytes, ...]
            self._pretokens[pt] += 1

    def _merge_once(self):
        pair_count: Counter[tuple[bytes, bytes]] = Counter()
        for pt in self._pretokens:
            # 统计所有相邻pair的出现频率
            for i in range(len(pt) - 1):
                pair_count[(pt[i], pt[i + 1])] += self._pretokens[pt]

        # 把频率最高的pair合并成一个新符号
        pair_most = max(pair_count, key=lambda k: (pair_count[k], k))
        self._merges.append(pair_most)
        bytes_merged = pair_most[0] + pair_most[1]
        self._vocab[len(self._vocab)] = bytes_merged

        # pretoken内部的合并
        for pt in self._pretokens.copy():
            pto = pt
            i = 0
            while i < len(pt) - 1:
                if pt[i : i + 2] == pair_most:
                    # 把所有该pair的出现都替换成新符号
                    pt = pt[:i] + (bytes_merged,) + pt[i + 2 :]
                i += 1
            if pt != pto:
                self._pretokens[pt] = self._pretokens.pop(pto)

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self._pre_tokenize()
        while len(self._vocab) < self._vocab_size:
            self._merge_once()
        return self._vocab, self._merges

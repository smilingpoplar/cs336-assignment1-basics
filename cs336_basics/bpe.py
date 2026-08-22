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
        self._special_tokens = special_tokens
        self._vocab: dict[int, bytes] = {c: bytes([c]) for c in range(256)}
        for st in special_tokens:
            self._vocab[len(self._vocab)] = st.encode("utf-8")
        self._pretokens: Counter[tuple[bytes, ...]] = Counter()
        self._merges: list[tuple[bytes, bytes]] = []
        self._pair_count: Counter[tuple[bytes, bytes]] = Counter()

    def _pre_tokenize(self):
        with open(self._input_path, encoding="utf-8") as f:
            text = f.read()
        st_pattern = "|".join([re.escape(st) for st in self._special_tokens])
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in re.split(st_pattern, text):
            for match in re.finditer(PAT, chunk):
                pt = match.group().encode("utf-8")
                pt = tuple(pt[i : i + 1] for i in range(len(pt)))  # tuple[bytes, ...]
                self._pretokens[pt] += 1

        for pt in self._pretokens:
            # 统计所有相邻pair的出现频率
            for i in range(len(pt) - 1):
                self._pair_count[(pt[i], pt[i + 1])] += self._pretokens[pt]

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

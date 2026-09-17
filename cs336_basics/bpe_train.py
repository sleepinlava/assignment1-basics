# import multiprocessing as mp
# import os
# from collections import defaultdict
# from typing import BinaryIO

# import regex as reg


# def str_to_bytes_func(input_list: list[str]) -> list[bytes]:
#     bytes_list: list[bytes] = [item.encode("utf-8") for item in input_list]
#     return bytes_list


# def find_chunk_boundaries(
#     file: BinaryIO,
#     desired_num_chunks: int,
#     split_special_tokens,  # 可以是 bytes，也可以是 list[bytes]
# ) -> list[int]:
#     """
#     将文件切分成可以独立计数的块。
#     支持多个 special token 作为分块边界。
#     如果边界重叠，返回的块数可能少于 desired_num_chunks。
#     """
#     # 兼容单个 bytes 或多个 bytes
#     if isinstance(split_special_tokens, bytes):
#         split_special_tokens = [split_special_tokens]
#     assert all(isinstance(t, bytes) for t in split_special_tokens), "每个 special token 必须表示为 bytestring"

#     # 获取文件大小
#     file.seek(0, os.SEEK_END)
#     file_size = file.tell()
#     file.seek(0)

#     chunk_size = file_size // desired_num_chunks

#     # 初始均匀分布的边界估计值
#     chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
#     chunk_boundaries[-1] = file_size

#     mini_chunk_size = 4096
#     max_token_len = max(len(t) for t in split_special_tokens)
#     overlap = max_token_len - 1  # 用于检测跨 mini chunk 的 special token

#     for bi in range(1, len(chunk_boundaries) - 1):
#         initial_position = chunk_boundaries[bi]
#         file.seek(initial_position)

#         tail = b""  # 上一次读取的末尾，用于跨块匹配
#         region_start = initial_position  # search_region 在文件中的起始绝对位置

#         while True:
#             mini_chunk = file.read(mini_chunk_size)
#             if mini_chunk == b"":
#                 # 到达 EOF，边界放到文件末尾
#                 chunk_boundaries[bi] = file_size
#                 break

#             search_region = tail + mini_chunk

#             # 在所有 special token 中找到最早出现的位置
#             earliest = -1
#             for tok in split_special_tokens:
#                 p = search_region.find(tok)
#                 if p != -1 and (earliest == -1 or p < earliest):
#                     earliest = p

#             if earliest != -1:
#                 chunk_boundaries[bi] = region_start + earliest
#                 break

#             # 没找到：准备下一次循环
#             # 保留末尾 overlap 个字节，防止 token 跨两个 mini_chunk 被漏掉
#             keep = min(len(search_region), overlap)
#             tail = search_region[-keep:] if keep > 0 else b""
#             region_start = region_start + len(search_region) - len(tail)

#     # 去掉重复边界，可能少于 desired_num_chunks
#     return sorted(set(chunk_boundaries))


# def pretoken_child_func(args):
#     """
#     act as the child function in mulitprocessing file
#     """
#     vocab_dict = defaultdict(int)
#     input_path, start_index, end_index, special_tokens, pat = args
#     with open(input_path, "rb") as f:
#         f.seek(start_index)
#         cut_chunks = f.read(end_index - start_index).decode("utf-8")
#         chunks = reg.split("|".join(map(reg.escape, special_tokens)), cut_chunks)
#         for corpus in chunks:
#             for token in reg.finditer(pat, corpus):
#                 tmp = token.group().encode("utf-8")
#                 vocab_dict[tuple(tmp[i : i + 1] for i in range(0, len(tmp)))] += 1
#     return dict(vocab_dict)


# def pretoken_par_func(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
#     # Match Pattern
#     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#     vocab_dict = defaultdict(int)
#     bytes_sp_tokens: list[bytes] = str_to_bytes_func(special_tokens)
#     num_process: int = mp.cpu_count()
#     index_chunks_list: list[int] = []

#     with open(input_path, "rb") as f:
#         index_chunks_list = find_chunk_boundaries(f, num_process, bytes_sp_tokens)

#     task_arg: list = []
#     # Q:为什么我们这里用 len(index_chunks_list) - 1
#     # A: 因为我们这里是一个有重叠的，也就是说 list_(n - 1)[i - 1] -> list_n[i] 是等价的
#     # 为了避免我们的终点超出我们的文件大小，so do that
#     for index in range(len(index_chunks_list) - 1):
#         start = index_chunks_list[index]
#         end = index_chunks_list[index + 1]
#         task_arg.append((input_path, start, end, special_tokens, PAT))

#     with mp.Pool(num_process) as p:
#         results = p.map(pretoken_child_func, task_arg)
#         # result是由多个并行处理后的结果向叠加起来的结果
#         # 最后需要一个一个取出结果并合并

#     for result in results:
#         for key, value in result.items():
#             vocab_dict[key] += value
#     return vocab_dict

# def merge_token(token, pair):
#     new_tokens = []
#     i = 0
#     while i < len(token):
#         if i < len(token) - 1 and token[i] == pair[0] and token[i + 1] == pair[1]:
#             new_tokens.append(token[i] + token[i + 1])
#             i += 2
#         else:
#             new_tokens.append(token[i])
#             i += 1
#     return tuple(new_tokens)


# def string_merge_loop(train_times, input_dict, result_list, record_list):
#     result_set = set(result_list)

#     for _ in range(train_times):
#         pair_freq = defaultdict(int)

#         # 1. 统计所有 pair 频次
#         for token, freq in input_dict.items():
#             for i in range(len(token) - 1):
#                 pair = (token[i], token[i + 1])
#                 pair_freq[pair] += freq

#         if not pair_freq:
#             break

#         # 2. 选 best_pair
#         best_pair = max(pair_freq, key=lambda p: (pair_freq[p], p))

#         # 3. 记录规则，只记录一次
#         merged = best_pair[0] + best_pair[1]
#         if merged not in result_set:
#             result_list.append(merged)
#             record_list.append(best_pair)
#             result_set.add(merged)

#         # 4. 遍历所有 token 合并，保留不包含 best_pair 的 token
#         new_input_dict = defaultdict(int)
#         for token, freq in input_dict.items():
#             new_token = merge_token(token, best_pair)
#             new_input_dict[new_token] += freq

#         input_dict = new_input_dict

# def bpe_train(
#     input_path: str, vocab_size: int, special_tokens: list[str]
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

#     bpe_train_table: list[bytes, ...] = []
#     bpe_record_list: list[bytes, ...] = []

#     index_sp = 0
#     index_bpe_table: dict[int:bytes] = {i: bytes([i]) for i in range(256)}
#     for item in special_tokens:
#         index_bpe_table[256 + index_sp] = item.encode("utf-8")
#         index_sp += 1

#     length_special_token: int = len(special_tokens)
#     num_merge: int = vocab_size - 256 - length_special_token

#     pretoken_result = pretoken_par_func(input_path, special_tokens)
#     string_merge_loop(num_merge, pretoken_result, bpe_train_table, bpe_record_list)

#     index_bpe = 0
#     for item in bpe_train_table:
#         index_bpe_table[256 + length_special_token + index_bpe] = item
#         index_bpe += 1

#     return (index_bpe_table, bpe_record_list)


# if __name__ == "__main__":
#     pass


import heapq
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO

import regex as reg

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PICKLE_VERSION = 1


# ============================================================
# 1. 通用：分块 / 预分词
# ============================================================
def str_to_bytes_func(input_list: list[str]) -> list[bytes]:
    return [item.encode("utf-8") for item in input_list]


def _adjust_to_utf8_boundary(file: BinaryIO, pos: int, file_size: int) -> int:
    if pos <= 0:
        return 0
    if pos >= file_size:
        return file_size
    file.seek(pos)
    while pos < file_size:
        b = file.read(1)
        if not b:
            return file_size
        if (b[0] & 0xC0) != 0x80:
            return pos
        pos += 1
    return file_size


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: bytes | list[bytes],
) -> list[int]:
    if isinstance(split_special_tokens, bytes):
        split_special_tokens = [split_special_tokens]
    assert all(isinstance(t, bytes) for t in split_special_tokens)

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        return [0]
    if desired_num_chunks <= 1:
        return [0, file_size]

    if not split_special_tokens:
        raw = [i * file_size // desired_num_chunks for i in range(desired_num_chunks + 1)]
        raw[-1] = file_size
        return sorted(set(_adjust_to_utf8_boundary(file, b, file_size) for b in raw))

    chunk_size = max(1, file_size // desired_num_chunks)
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    max_token_len = max(len(t) for t in split_special_tokens)
    overlap = max_token_len - 1

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        tail = b""
        region_start = initial_position

        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            search_region = tail + mini_chunk
            earliest = -1
            for tok in split_special_tokens:
                p = search_region.find(tok)
                if p != -1 and (earliest == -1 or p < earliest):
                    earliest = p
            if earliest != -1:
                chunk_boundaries[bi] = region_start + earliest
                break
            keep = min(len(search_region), overlap)
            tail = search_region[-keep:] if keep > 0 else b""
            region_start = region_start + len(search_region) - len(tail)

    return sorted(set(chunk_boundaries))


def pretoken_child_func(args):
    input_path, start_index, end_index, special_tokens, pat = args
    vocab_dict: dict[tuple[bytes, ...], int] = defaultdict(int)

    with open(input_path, "rb") as f:
        f.seek(start_index)
        raw = f.read(end_index - start_index)

    cut_chunks = raw.decode("utf-8", errors="ignore")

    if special_tokens:
        split_pattern = "|".join(map(reg.escape, special_tokens))
        chunks = reg.split(split_pattern, cut_chunks)
    else:
        chunks = [cut_chunks]

    for corpus in chunks:
        for token in reg.finditer(pat, corpus):
            tmp = token.group().encode("utf-8")
            vocab_dict[tuple(tmp[i : i + 1] for i in range(len(tmp)))] += 1

    return dict(vocab_dict)


def pretoken_par_func(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    bytes_sp_tokens = str_to_bytes_func(special_tokens)
    num_process = mp.cpu_count()

    with open(input_path, "rb") as f:
        index_chunks_list = find_chunk_boundaries(f, num_process, bytes_sp_tokens)

    task_arg = []
    for i in range(len(index_chunks_list) - 1):
        s, e = index_chunks_list[i], index_chunks_list[i + 1]
        if s < e:
            task_arg.append((input_path, s, e, special_tokens, PAT))

    vocab_dict: dict[tuple[bytes, ...], int] = defaultdict(int)
    if not task_arg:
        return dict(vocab_dict)

    actual_num_process = min(num_process, len(task_arg))
    ctx = mp.get_context()
    with ctx.Pool(actual_num_process) as p:
        results = p.map(pretoken_child_func, task_arg)

    for result in results:
        for key, value in result.items():
            vocab_dict[key] += value
    return dict(vocab_dict)


# ============================================================
# 2. 加速版 BPE 训练器
# ============================================================
class _HeapItem:
    __slots__ = ("neg_freq", "pair")

    def __init__(self, freq: int, pair: tuple[bytes, bytes]):
        self.neg_freq = -freq
        self.pair = pair

    def __lt__(self, other: "_HeapItem") -> bool:
        if self.neg_freq != other.neg_freq:
            return self.neg_freq < other.neg_freq
        return self.pair > other.pair


def _merge_token(token: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    new_tokens: list[bytes] = []
    i = 0
    n = len(token)
    while i < n:
        if i < n - 1 and token[i] == pair[0] and token[i + 1] == pair[1]:
            new_tokens.append(token[i] + token[i + 1])
            i += 2
        else:
            new_tokens.append(token[i])
            i += 1
    return tuple(new_tokens)


class BPETrainer:
    __slots__ = ("token_freq", "pair_freq", "pair_to_tokens", "heap")

    def __init__(self, token_freq: dict[tuple[bytes, ...], int]):
        self.token_freq: dict[tuple[bytes, ...], int] = dict(token_freq)
        self.pair_freq: dict[tuple[bytes, bytes], int] = {}
        self.pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

        for token, freq in self.token_freq.items():
            for i in range(len(token) - 1):
                p = (token[i], token[i + 1])
                self.pair_freq[p] = self.pair_freq.get(p, 0) + freq
                self.pair_to_tokens.setdefault(p, set()).add(token)

        self.heap: list[_HeapItem] = [_HeapItem(freq, p) for p, freq in self.pair_freq.items()]
        heapq.heapify(self.heap)

    def _pop_best_pair(self):
        heap = self.heap
        pair_freq = self.pair_freq
        while heap:
            item = heap[0]
            p = item.pair
            cur = pair_freq.get(p, 0)
            if cur != -item.neg_freq:
                heapq.heappop(heap)
                continue
            heapq.heappop(heap)
            return p, cur
        return None, 0

    def step(self):
        best_pair, _ = self._pop_best_pair()
        if best_pair is None:
            return None

        token_freq = self.token_freq
        pair_freq = self.pair_freq
        pair_to_tokens = self.pair_to_tokens
        push = heapq.heappush
        heap = self.heap

        affected = list(pair_to_tokens.get(best_pair, ()))
        for token in affected:
            freq = token_freq.get(token, 0)
            if freq == 0:
                continue

            new_token = _merge_token(token, best_pair)
            if new_token == token:
                continue

            n = len(token)
            for i in range(n - 1):
                p = (token[i], token[i + 1])
                new_f = pair_freq.get(p, 0) - freq
                if new_f <= 0:
                    pair_freq.pop(p, None)
                    pair_to_tokens.pop(p, None)
                else:
                    pair_freq[p] = new_f
                    s = pair_to_tokens.get(p)
                    if s is not None:
                        s.discard(token)
                        if not s:
                            pair_to_tokens.pop(p, None)
                    push(heap, _HeapItem(new_f, p))

            del token_freq[token]
            token_freq[new_token] = token_freq.get(new_token, 0) + freq

            m = len(new_token)
            for i in range(m - 1):
                p = (new_token[i], new_token[i + 1])
                new_f = pair_freq.get(p, 0) + freq
                pair_freq[p] = new_f
                pair_to_tokens.setdefault(p, set()).add(new_token)
                push(heap, _HeapItem(new_f, p))

        return best_pair

    def train(self, num_merges: int) -> list[tuple[bytes, bytes]]:
        merges: list[tuple[bytes, bytes]] = []
        for _ in range(num_merges):
            bp = self.step()
            if bp is None:
                break
            merges.append(bp)
        return merges


# ============================================================
# 3. 持久化模型
# ============================================================
class BPEModel:
    """
    可 pickle 的 BPE 模型。

    存储的字段：
        vocab:          dict[int, bytes]           id -> 字节串
        merges:         list[tuple[bytes, bytes]]  合并顺序
        special_tokens: list[str]                  特殊标记
        pat:            str                        预分词正则
        version:        int                        pickle 格式版本

    派生字段（惰性构建，不进 pickle）：
        _token_to_id:   dict[bytes, int]
        _merge_ranks:   dict[tuple[bytes,bytes], int]
    """

    __slots__ = ("vocab", "merges", "special_tokens", "pat", "version", "_token_to_id", "_merge_ranks")

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        pat: str = PAT,
        version: int = PICKLE_VERSION,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.pat = pat
        self.version = version
        self._token_to_id = None
        self._merge_ranks = None

    # -------- 派生表惰性构建 --------
    @property
    def token_to_id(self) -> dict[bytes, int]:
        if self._token_to_id is None:
            self._token_to_id = {v: k for k, v in self.vocab.items()}
        return self._token_to_id

    @property
    def merge_ranks(self) -> dict[tuple[bytes, bytes], int]:
        if self._merge_ranks is None:
            self._merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        return self._merge_ranks

    # -------- pickle 时排除派生表 --------
    def __getstate__(self):
        return {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "pat": self.pat,
            "version": self.version,
        }

    def __setstate__(self, state):
        self.vocab = state["vocab"]
        self.merges = state["merges"]
        self.special_tokens = state.get("special_tokens", [])
        self.pat = state.get("pat", PAT)
        self.version = state.get("version", PICKLE_VERSION)
        self._token_to_id = None
        self._merge_ranks = None

    # -------- 保存 / 加载 --------
    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 先写临时文件再 rename，避免写一半崩溃导致文件损坏
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "BPEModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} 里不是 BPEModel，而是 {type(obj)}")
        return obj

    # -------- 编码 / 解码（方便后续直接用） --------
    def _bpe(self, piece: tuple[bytes, ...]) -> tuple[bytes, ...]:
        """按 merge_ranks 从小到大反复合并，得到最终 tokens。"""
        ranks = self.merge_ranks
        if not ranks or len(piece) < 2:
            return piece
        # 反复找 rank 最小的相邻 pair 合并
        while len(piece) > 1:
            best_rank = None
            best_i = -1
            for i in range(len(piece) - 1):
                r = ranks.get((piece[i], piece[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_i == -1:
                break
            piece = piece[:best_i] + (piece[best_i] + piece[best_i + 1],) + piece[best_i + 2 :]
        return piece

    def encode(self, text: str) -> list[int]:
        token_to_id = self.token_to_id
        ids: list[int] = []

        # 1) 按 special token 切分，保留分隔符
        if self.special_tokens:
            pat = "(" + "|".join(map(reg.escape, self.special_tokens)) + ")"
            chunks = reg.split(pat, text)
        else:
            chunks = [text]

        special_set = set(self.special_tokens)
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in special_set:
                ids.append(token_to_id[chunk.encode("utf-8")])
                continue
            for m in reg.finditer(self.pat, chunk):
                raw = m.group().encode("utf-8")
                piece = tuple(bytes([b]) for b in raw)
                for b in self._bpe(piece):
                    ids.append(token_to_id[b])
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")


# ============================================================
# 4. 训练入口
# ============================================================
# def bpe_train(
#     input_path: str,
#     vocab_size: int,
#     special_tokens: list[str],
#     save_path: str | os.PathLike | None = None,
#     *args,
#     **kwargs,
# ) -> BPEModel:
def bpe_train_test(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    训练 BPE，返回 BPEModel。
    如果给了 save_path，训练完自动保存一份 pickle。
    """
    if vocab_size < 256 + len(special_tokens):
        raise ValueError(f"vocab_size={vocab_size} 太小，至少要 {256 + len(special_tokens)}")

    index_bpe_table: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, tok in enumerate(special_tokens):
        index_bpe_table[256 + i] = tok.encode("utf-8")

    length_special_token = len(special_tokens)
    num_merge = vocab_size - 256 - length_special_token

    pretoken_result = pretoken_par_func(input_path, special_tokens)
    trainer = BPETrainer(pretoken_result)
    merges = trainer.train(num_merge)

    for i, pair in enumerate(merges):
        index_bpe_table[256 + length_special_token + i] = pair[0] + pair[1]

    model = BPEModel(
        vocab=index_bpe_table,
        merges=merges,
        special_tokens=list(special_tokens),
        pat=PAT,
    )

    # if save_path is not None:
    # model.save(save_path)

    # return model
    return model.vocab, model.merges


def bpe_train(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    save_path: str | os.PathLike | None = None,
    *args,
    **kwargs,
) -> BPEModel:
    """
    训练 BPE，返回 BPEModel。
    如果给了 save_path，训练完自动保存一份 pickle。
    """
    if vocab_size < 256 + len(special_tokens):
        raise ValueError(f"vocab_size={vocab_size} 太小，至少要 {256 + len(special_tokens)}")

    index_bpe_table: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, tok in enumerate(special_tokens):
        index_bpe_table[256 + i] = tok.encode("utf-8")

    length_special_token = len(special_tokens)
    num_merge = vocab_size - 256 - length_special_token

    pretoken_result = pretoken_par_func(input_path, special_tokens)
    trainer = BPETrainer(pretoken_result)
    merges = trainer.train(num_merge)

    for i, pair in enumerate(merges):
        index_bpe_table[256 + length_special_token + i] = pair[0] + pair[1]

    model = BPEModel(
        vocab=index_bpe_table,
        merges=merges,
        special_tokens=list(special_tokens),
        pat=PAT,
    )

    if save_path is not None:
        model.save(save_path)

    return model


# ============================================================
# 5. 使用示例
# ============================================================
if __name__ == "__main__":
    # # ---- 训练并保存 ----
    # model = bpe_train(
    #     input_path="/home/bker/cs336_2026/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"]
    #     save_path="/home/bker/cs336_2026/assignment1-basics/checkpoints/bpe_tinystory.pkl",
    # )
    pass
    # ---- 之后只用加载 ----
    # model = BPEModel.load("checkpoints/bpe_1000.pkl")

    # ids = model.encode("Hello world! <|endoftext|> 你好")
    # print(ids)
    # print(model.decode(ids))

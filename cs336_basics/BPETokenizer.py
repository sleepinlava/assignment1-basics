import multiprocessing as mp
import os
from collections import defaultdict
from typing import BinaryIO

import regex as re


def str_to_bytes_func(input_list: list[str]) -> list[bytes]:
    bytes_list: list[bytes] = [item.encode("utf-8") for item in input_list]
    return bytes_list


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens,  # 可以是 bytes，也可以是 list[bytes]
) -> list[int]:
    """
    将文件切分成可以独立计数的块。
    支持多个 special token 作为分块边界。
    如果边界重叠，返回的块数可能少于 desired_num_chunks。
    """
    # 兼容单个 bytes 或多个 bytes
    if isinstance(split_special_tokens, bytes):
        split_special_tokens = [split_special_tokens]
    assert all(isinstance(t, bytes) for t in split_special_tokens), "每个 special token 必须表示为 bytestring"

    # 获取文件大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 初始均匀分布的边界估计值
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    max_token_len = max(len(t) for t in split_special_tokens)
    overlap = max_token_len - 1  # 用于检测跨 mini chunk 的 special token

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        tail = b""  # 上一次读取的末尾，用于跨块匹配
        region_start = initial_position  # search_region 在文件中的起始绝对位置

        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                # 到达 EOF，边界放到文件末尾
                chunk_boundaries[bi] = file_size
                break

            search_region = tail + mini_chunk

            # 在所有 special token 中找到最早出现的位置
            earliest = -1
            for tok in split_special_tokens:
                p = search_region.find(tok)
                if p != -1 and (earliest == -1 or p < earliest):
                    earliest = p

            if earliest != -1:
                chunk_boundaries[bi] = region_start + earliest
                break

            # 没找到：准备下一次循环
            # 保留末尾 overlap 个字节，防止 token 跨两个 mini_chunk 被漏掉
            keep = min(len(search_region), overlap)
            tail = search_region[-keep:] if keep > 0 else b""
            region_start = region_start + len(search_region) - len(tail)

    # 去掉重复边界，可能少于 desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretoken_child_func(args):
    """
    act as the child function in mulitprocessing file
    """
    vocab_dict = defaultdict(int)
    input_path, start_index, end_index, special_tokens, pat = args
    with open(input_path, "rb") as f:
        f.seek(start_index)
        cut_chunks = f.read(end_index - start_index).decode("utf-8")
        chunks = re.split("|".join(map(re.escape, special_tokens)), cut_chunks)
        for corpus in chunks:
            for token in re.finditer(pat, corpus):
                tmp = token.group().encode("utf-8")
                vocab_dict[tuple(tmp[i : i + 1] for i in range(0, len(tmp)))] += 1
    return dict(vocab_dict)


def pretoken_par_func(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    # Match Pattern
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    vocab_dict = defaultdict(int)
    bytes_sp_tokens: list[bytes] = str_to_bytes_func(special_tokens)
    num_process: int = mp.cpu_count()
    index_chunks_list: list[int] = []

    with open(input_path, "rb") as f:
        index_chunks_list = find_chunk_boundaries(f, num_process, bytes_sp_tokens)

    task_arg: list = []
    # Q:为什么我们这里用 len(index_chunks_list) - 1
    # A: 因为我们这里是一个有重叠的，也就是说 list_(n - 1)[i - 1] -> list_n[i] 是等价的
    # 为了避免我们的终点超出我们的文件大小，so do that
    for index in range(len(index_chunks_list) - 1):
        start = index_chunks_list[index]
        end = index_chunks_list[index + 1]
        task_arg.append((input_path, start, end, special_tokens, PAT))

    # ctx = mp.get_context("forkserver")
    with mp.Pool(num_process) as p:
        results = p.map(pretoken_child_func, task_arg)
        # result是由多个并行处理后的结果向叠加起来的结果
        # 最后需要一个一个取出结果并合并

    for result in results:
        for key, value in result.items():
            vocab_dict[key] += value
    return vocab_dict


# def pretoken(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:

#     vocab_dict = defaultdict(int)
#     bytes_sp_tokens: list[bytes] = str_to_bytes_func(special_tokens)
#     index_chunks_list: list[int] = []
#     num_processes: int = mp.cpu_count()
#     with open(input_path, "r+", encoding="utf-8") as f:
#         with mp.Pool(num_processes) as p:
#             index_chunks_list = find_chunk_boundaries(f, num_processes, bytes_sp_tokens)
#             tmp_test = f.read()
#             # map() 是 Python 的一个内置高阶函数。它的核心含义是：将指定的函数依次作用于可迭代对象（如列表、元组）中的每一个元素，并返回一个迭代器。
#             long_corpus = re.split("|".join(map(re.escape, special_tokens)), tmp_test)
#             for corpus in long_corpus:
#                 for token in re.finditer(PAT, corpus):
#                     tmp = token.group().encode("utf-8")
#                     vocab_dict[tuple(tmp[i : i + 1] for i in range(0, len(tmp)))] += 1
#     return vocab_dict


def get_freqs_bytes_dict(input_dict: dict[bytes, ...]) -> dict[tuple[bytes, ...], int]:
    pair_freq = defaultdict(int)
    for token, freq in input_dict.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_freq[pair] += freq
    return pair_freq


def merge_Dict_by_TwoBytes(
    token, pair, result_list: list[bytes, ...], record_list: list[bytes, ...]
) -> tuple[list[bytes, ...]]:
    new_tokens = []  # list -> tuple
    i = 0
    # match the tokens in courpus
    while i < len(token):
        if (
            # edge condition judgement
            i < len(token) - 1
            and token[i + 1] == pair[1]
            # right byte == left pair byte
            and token[i] == pair[0]
            # left byte == righte pair byte
            # double bytes pair in once merge, so we choose the len(pair) == 2
        ):
            new_tokens.append(token[i] + token[i + 1])
            i += 2
        else:
            # add unmatched tokens into string
            new_tokens.append(token[i])
            i += 1
    if (pair[0] + pair[1]) not in result_list:
        result, record = pair[0] + pair[1], (pair[0], pair[1])
        result_list.append(result)
        record_list.append(record)
    # 因为我们的best_pair是从我们的tokens中通过max选出来的，
    # 必然存在于tokens，所以最后记一次数就可以了
    return tuple(new_tokens)


def string_merge_loop(
    train_times: int,
    input_dict: dict[tuple[bytes, ...], int],
    result_list: list[bytes, ...],
    record_list: list[bytes, ...],
) -> None:
    for _ in range(train_times):
        original_string_freqs = get_freqs_bytes_dict(input_dict)
        # best_pair = max(original_string_freqs, key=lambda value : original_string_freqs)
        best_pair = max(original_string_freqs, key=lambda pair: (original_string_freqs[pair], pair))
        merged_string = defaultdict(int)
        for tokens, freqs in input_dict.items():
            # do a check loop in original byte string, and fix it
            merged_pair = merge_Dict_by_TwoBytes(tokens, best_pair, result_list, record_list)
            merged_string[merged_pair] += freqs
        input_dict = merged_string
        # **need update the data in the orignal byte string**
    # return input_dict.keys()
    return


def bpe_train(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    bpe_train_table: list[bytes, ...] = []
    bpe_record_list: list[bytes, ...] = []

    index_sp = 0
    index_bpe_table: dict[int:bytes] = {i: bytes([i]) for i in range(256)}
    for item in special_tokens:
        index_bpe_table[256 + index_sp] = item.encode("utf-8")
        index_sp += 1

    length_special_token: int = len(special_tokens)
    num_merge: int = vocab_size - 256 - length_special_token

    pretoken_result = pretoken_par_func(input_path, special_tokens)
    string_merge_loop(num_merge, pretoken_result, bpe_train_table, bpe_record_list)

    index_bpe = 0
    for item in bpe_train_table:
        index_bpe_table[256 + length_special_token + index_bpe] = item
        index_bpe += 1

    return (index_bpe_table, bpe_record_list)


if __name__ == "__main__":
    # Path about train/valid_dataset_path
    train_dataset_path = os.path.expanduser("~/cs336_2026/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")
    valid_dataset_path = os.path.expanduser("~/cs336_2026/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt")
    test_tiny_stroy_path = os.path.expanduser("~/cs336_2026/assignment1-basics/data/tiny_test_data.txt")

    bpe_train_table: list = []
    bpe_record_list: list = []
    special_tokens_list: list[str] = ["<|endoftext|>", "<|pad|>", "<|unk|>"]

    # with mp.Pool(8) as p:
    #     print(bpe_train(test_tiny_stroy_path, 500, special_tokens_list))
    #     p.join()
    #     p.close()
    # How to use it?
    # in main()

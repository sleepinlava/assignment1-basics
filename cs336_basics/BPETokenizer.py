import os
from collections import defaultdict

import regex as re

# Match Pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries():
    pass


def pretoken(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    vocab_dict = defaultdict(int)
    with open(input_path, "r+", encoding="utf-8") as f:
        tmp_test = f.read()
        # map() 是 Python 的一个内置高阶函数。它的核心含义是：将指定的函数依次作用于可迭代对象（如列表、元组）中的每一个元素，并返回一个迭代器。
        long_corpus = re.split("|".join(map(re.escape, special_tokens)), tmp_test)
        for corpus in long_corpus:
            for token in re.finditer(PAT, corpus):
                tmp = token.group().encode("utf-8")
                vocab_dict[tuple(tmp[i : i + 1] for i in range(0, len(tmp)))] += 1
    return vocab_dict


def get_freqs_bytes_dict(input_dict: dict[bytes, ...]) -> dict[tuple[bytes, ...], int]:
    pair_freq = defaultdict(int)
    for token, freq in input_dict.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_freq[pair] += freq
    return pair_freq


def merge_Dict_by_TwoBytes(token, pair, result_list: list[bytes, ...], record_list: list[bytes, ...]) -> tuple[list[bytes, ...]]:
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


def string_merge_loop(train_times: int, input_dict: dict[tuple[bytes, ...], int], result_list: list[bytes, ...], record_list: list[bytes, ...]) -> None:
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

def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str])-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    bpe_train_table: list[bytes, ...] = []
    bpe_record_list: list[bytes, ...] = []


    index_sp = 0
    index_bpe_table : dict[int : bytes] = {i : bytes([i]) for i in range(256)}
    for item in special_tokens:
        index_bpe_table[256 + index_sp] = item.encode('utf-8')
        index_sp += 1

    length_special_token : int = len(special_tokens)
    num_merge : int = vocab_size - 256 - length_special_token

    pretoken_result = pretoken(input_path, special_tokens)
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

    print(bpe_train(test_tiny_stroy_path, 400, special_tokens_list))


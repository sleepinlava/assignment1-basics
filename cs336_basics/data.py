import os
import typing

import numpy as np
import numpy.typing as npt
import torch

np.random.seed(42)


def data_load(
    array: npt.NDArray | np.memmap, batch_size: int, context_length: int, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:

    available_array_length = array.size - context_length - 1

    index = np.random.randint(0, available_array_length + 1, size=batch_size)

    sample_list = [array[idx : idx + context_length] for idx in index]
    label_list = [array[idx + 1 : idx + context_length + 1] for idx in index]

    # 这里为什么要用np.stack()？
    # 这里虽然看起来shape和我们目标的shape一样
    # 但是它仍然是一个list,而没有shape
    # 或者换句话来说，我们需要一个合理的方法将list
    # 转换为np类型

    sample_tensor = torch.from_numpy(np.stack(sample_list)).to(device)
    label_tensor = torch.from_numpy(np.stack(label_list)).to(device)

    return (sample_tensor, label_tensor)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteratin: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    dict_information: dict = {}

    dict_information["obj_model"] = model.state_dict()
    dict_information["obj_opt"] = optimizer.state_dict()
    dict_information["iteration"] = iteratin
    torch.save(dict_information, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    dict_information = torch.load(src)
    model.load_state_dict(dict_information["obj_model"])
    optimizer.load_state_dict(dict_information["obj_opt"])
    iteration = dict_information["iteration"]
    return iteration


if __name__ == "__main__":
    pass
    # print(randint(0, 10))
    # list_np: list = [i for i in range(1, 17)]
    # a = np.asarray(list_np).reshape(4, 4)
    # print(a)
    # # pass
    # b = np.asarray(list_np)
    # print(b)
    # print(len(b))
    # print(b.size)
    # a = np.arange(16)
    # print(a[1:5])
    # print(5 % 2)
    # test_list = np.arange(0, 10)
    # lr = np.random.randint(0, 9, size=10)
    # rr = lr + 1
    # print(lr)
    # for _ in rr:
    #     print(test_list[_])
    a = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    print(a)
    print(np.stack(a))

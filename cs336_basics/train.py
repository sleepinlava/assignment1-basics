import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from cs336_basics.bpe_train import BPEModel
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data import data_load, load_checkpoint, save_checkpoint
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.softmax import softmax
from cs336_basics.transformer import transformer_lm


@torch.no_grad()
def evaluate_fn(model: nn.Module, data_array: np.memmap, hyper_params: dict):
    model.eval()

    batch_size = hyper_params["batch_size"]
    context_length = hyper_params["context_length"]
    eval_device = hyper_params["device"]

    total_loss = -1e5
    correct = -1e5
    total = -1e5

    x, y = data_load(data_array, batch_size, context_length, eval_device)

    logits = model(x)
    loss = cross_entropy(logits, y)

    preds = softmax(logits, dim=-1).argmax(dim=-1)

    result = torch.eq(preds, y).sum()


def full_train(
    data_load_path: os.PathLike,
    output_path: os.PathLike,
    train_times: int,
    model: nn.Module,
    hyper_params: dict,
    model_load_path=None,
    load_from_former=False,
) -> None:

    def single_train(
        model: nn.Module,
        hyper_params: dict,
        optimizer: optim.Optimizer,
        data_array: np.memmap,
        global_train_times: int,
    ):

        vocab_size = hyper_params["vocab_size"]
        context_length = hyper_params["context_length"]
        train_device = hyper_params["device"]
        max_lr = hyper_params["max_lr"]
        min_lr = hyper_params["min_lr"]
        t_w = hyper_params["t_w"]
        t_c = hyper_params["t_c"]
        l2_norm_max = hyper_params["hyper_params"]
        # 声明我们所需要用到的超参数

        x, y = data_load(data_array, vocab_size, context_length, train_device)
        ad_lr = get_lr_cosine_schedule(global_train_times, max_lr, min_lr, t_w, t_c)

        for group in optimizer.param_groups:
            group["lr"] = ad_lr
        # 优化器内部参数储存在param_groups
        # 我们这里使用了正弦调整学习率，因此需要更新

        optimizer.zero_grad()

        logistic = model(x)
        loss = cross_entropy(logistic, y)
        loss.backward()
        gradient_clipping(model.parameters(), l2_norm_max)
        optimizer.step()
        # 按照参数根据梯度更新

        return loss
        # 返回交叉熵损失，当loss降低时储存下来
        #

    model.train()

    run_times = 0
    max_lr = hyper_params["max_lr"]
    weight_decay = hyper_params["weight_decay"]
    betas = hyper_params["betas"]
    eps = hyper_params["eps"]

    data_list = np.memmap(data_load_path, dtype=np.uint32, mode="r")

    opt = AdamW(model.parameters, max_lr, weight_decay, betas, eps)
    # 想一想正弦调整学习率的目的是什么

    if load_from_former == True:
        if model_load_path is not None:
            run_times = load_checkpoint(model_load_path, model, opt)
        else:
            raise ValueError("No Model Path was inputed")

    for t in range(train_times - run_times):
        single_loss = single_train(model, hyper_params, opt, data_list, t + run_times)
        print(f"epoch is{t}, and the loss is {single_loss}")
        # 使用t + run_times的目的
        # 因为我们这里实际上要实现在整个周期上进行学习率的正弦调节
        # 如果只使用t的话会丢失前面学习率的调节

    save_checkpoint(model, opt, train_times, output_path)


if __name__ == "__main__":
    hyper_params: dict = {
        "vocab_size": int,
        "context_length": int,
        "num_layers": int,
        "d_model": int,
        "num_heads": int,
        "d_ff": int,
        "theta": float,
        "train_times": int,
        "max_lr": float,
        "min_lr": float,
        "t_w": int,
        "t_c": int,
        "l2_norm": float,
        "weight_decay": float,
        "betas": tuple,
        "eps": float,
        "device": torch.device,
    }

    # pass
    # enc = tiktoken.get_en        gradient_clipping(model.parameters(), 1)
    # with open(
    #     "/home/sleepinlava/Projects/python/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
    #     encoding="utf-8",
    # ) as f:
    #     text = f.read()
    # # ids = enc.encode_ordinary(text)
    # bpe_model = BPEModel.load(
    #     "/home/sleepinlava/Projects/python/cs336/assignment1-basics/checkpoints/bpe_tinystory.pkl"
    # )
    # print(model.vocab.items())
    # ids = model.encode(text)
    # arr = np.array(ids, dtype=np.uint32)
    # arr.tofile("test.bin")
    # # print(arr[0:1000])
    # print(arr.max(), arr.shape, arr.size)

    plt.ion()  # plt的交互模式
    fig, ax = plt.subplots()
    train_losses = []
    train_steps = []

    (line,) = ax.plot([], [], label="Train Loss")
    ax.set_xlabel("train steps")
    ax.set_ylabel("train loss")
    ax.grid(True)
    ax.legend()

    model = transformer_lm(9999, 1024, 12, 768, 12, 2048, 0.01)
    data = np.memmap("/home/sleepinlava/Projects/python/cs336/assignment1-basics/data/test.bin", dtype=np.uint32)

    # print(data[10:50])

    device = torch.device("cpu")
    x, y = data_load(data, 10, 16, device)

    opt = AdamW(model.parameters(), 15, 0.1, (0.9, 0.99), 1e-5)

    model.train()

    for _ in range(60):
        ad_lr = get_lr_cosine_schedule(_, 10, 0.9, 10, 30)
        for group in opt.param_groups:
            group["lr"] = ad_lr

        opt.zero_grad()

        logistic = model(x)
        loss = cross_entropy(logistic, y)
        loss.backward()
        # loss.backward()这里计算出来梯度后，就需要对可能大的梯度进行裁减
        gradient_clipping(model.parameters(), 1)
        opt.step()
        # 按照参数根据梯度更新

        with torch.no_grad():
            train_steps.append(_)
            train_losses.append(loss.item())

            line.set_data(train_steps, train_losses)

            # 动态调整坐标轴
            ax.relim()
            ax.autoscale_view()

            # refresh fig

            fig.canvas.draw_idle()
            plt.pause(0.01)
            plt.savefig("./loss.png")

        #     plt.xlabel("x - train_times")
        #     plt.ylabel("y - train_loss")
        #     plt.plot(_, loss.item())
        #     plt.grid(visible=True)
        #     plt.show()

        print(f"this is {_} times, and this epochs loss is {loss}")

    # save_checkpoint(model, opt, 50, "./test.pth")

    # model.state_dict()
    torch.save(model.state_dict(), "test.pth")

    # logits = model(x)

    # print(logits.shape, logits.size)

    # print(x.dtype, y.dtype)
    # print(data.dtype, data.shape, data.size)
    # print(data.max())
    # print(x[0:100])
    # print(x.shape, x.dtype)

    # logtis = model(x.to(torch.LongTensor()))
    # pre = softmax(logtis, dim=-1).argmax(dim=-1)

    # print(f"{pre}\n{logtis}")
    # print(pre.shape, pre.dtype)
    # print(y.shape, y.dtype)
    # print(torch.tensor(pre == y))
    # print([pre == y])
    # print(bpe_model.vocab.get())
    # a = ([1, 2, 3, 4], [2, 3, 4, 5])
    # for x, y in a:
    #     print(x, y)
    torch.arange(0, 16, -1)

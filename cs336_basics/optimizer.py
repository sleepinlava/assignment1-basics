import math
from collections.abc import Callable
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3) -> None:
        # 这行代码声明了一个名为 closure 的可选参数，
        # 类型可以是任意可调用对象或 None，默认是 None。
        # 它让 step 方法可以灵活地接受一个用于计算损失的闭包函数。
        if lr < 0:
            raise ValueError(f"Invaild learning rage:{lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # get 学习率
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps) -> None:

        if lr < 0:
            raise ValueError(f"Ivaild learning  range:{lr}")

        defaults = {
            "lr": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "epsilon": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    # 语法糖，就是在定义好的类或者继承的类的函数的基础上，进行自己的操作
    # 而机器会帮你翻译成为更加底层的语言进行操作
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            eplison = group["epsilon"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # if state["m"] is not None and state["v"] is not None:
                #     m = torch.zeros(p.shape)
                #     # version-1 一阶矩估计 这里是不是没保存下来？ right
                #     v = torch.zeros(p.shape)
                # # version-2 二阶矩估计 这里同上

                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                # 为什么这里要用t的次方
                # 为了进行偏差修正 m_t = beta_1 * m_t-1 + (1 - beta_1) * grad_t
                # 可计算得到 m_t 的数学期望
                # E(m_t) = E(g) * (1 - beta_1) * beta_1.sum(0 ~ t-1) = E(g) * (1 - beta_1 ** t)
                # 在初期, 因为m_0 = 0，所以E(m_1) = E(g) * (1 - beta_1)
                # 若beta_1 = 0.9，所以m1的期望只有梯度的10%，所以在初期梯度被严重缩小了
                # with torch.no_grad():
                grad = p.grad

                lr_new = lr * ((1 - beta_2**t) ** 0.5) / (1 - beta_1**t)

                # new_p = (1 - lr * weight_decay) * p
                p.copy_((1 - lr * weight_decay) * p)
                # 为什么不这样写 p = (1 - lr * weight_decay) * p
                # 因为python会给后面的计算结果分配给新的地址，然后前面的地址被更改到新的地址
                # 这个过程中导致实际上我们的para更本没有更新，而是在优化器内存地址外面发生变动
                # 如果想要原地修改可用它们内部的定义的相关方法

                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * ((grad) ** 2)

                # 在数学中，m, v是递归定义的，它的当前值依赖于上一个时刻的数值
                # adj_p = p - lr * (m) / (v + eplison) ** 0.5
                p.copy_(p - lr_new * ((m) / (v**0.5 + eplison)))

                lr = lr_new

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def get_lr_cosine_schedule(t: int, lr_max: float, lr_min: float, t_w: int, t_c: int) -> float:
    result_lr = 0
    if t < t_w:
        result_lr = t / t_w * lr_max
    elif t_w <= t <= t_c:
        result_lr = lr_min + 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (lr_max - lr_min)
    else:
        result_lr = lr_min

    return result_lr


@torch.no_grad()
def gradient_clipping(params: Iterable[torch.nn.Parameter], l2_norm_max: float):
    eps: float = 10**-6
    params_list: list = [v for v in params if v.grad is not None]

    l2_norm_all = torch.sqrt(sum(p.grad.pow(2).sum() for p in params_list))

    fact = l2_norm_max / (l2_norm_all + eps)

    if l2_norm_all > l2_norm_max:
        for p in params_list:
            p.grad.copy_(fact * p.grad)


# for p in params_list:
#     _ = fact * p.grad
#     p.grad.copy_(_)


#     for v in params:
#
#
#
#     # if l2_norm < l2_norm_max:
#     #     result = (l2_norm_max) / (l2_norm + eps)
#     # else:
#     #     result = l2_norm
#     return params


# torch.norm() -> calculate the tensor norm, implicit return L2 norm, explicit L1 norm, using `torch.norm(p = 1)`


if __name__ == "__main__":
    pass

    #     import torch
    #
    #     # 1. 模拟一个模型参数
    #     para = torch.nn.Parameter(torch.ones(3))
    #     params = [para]
    #
    #     # 2. 模拟你的错误代码
    #     for p in params:
    #         print(f"循环开始时，p的地址: {id(p)}, para的地址: {id(para)}")  # 地址相同
    #
    #         # 执行错误的非原位操作
    #         p = p * 2  # p 的标签被撕下来，贴到了新张量上
    #
    #         print(f"执行 p = p * 2 后，p的地址: {id(p)}")  # 地址变了！
    #         print(f"此时 para 的值: {para.data}")  # 依然是 1, 1, 1，没变！
    #
    #     print(f"循环结束后，para 的最终值: {para.data}")  # 依然是 1, 1, 1
    #
    #     a = 5
    #     print(f"before a idx{id(a)}")
    #     a = a * 2
    #     print(f"after a idx{id(a)}")
    #     print(10 - 6)
    # a = torch.rand(5)
    # b = torch.rand(5)
    # print(a , b)
    # print(torch.stack([a,b]))

    #     # 错误写法：generator
    #     def bad(params):
    #         params = (p for p in params if p.grad is not None)
    #         if not params:
    #             print("空！")
    #         total = sum(p.grad.sum() for p in params)   # 耗尽
    #         count = sum(1 for p in params)              # 这里是 0
    #         print(f"处理了 {count} 个参数")
    #
    #     # 正确写法：list
    #     def good(params):
    #         params = [p for p in params if p.grad is not None]
    #         if not params:
    #             print("空！")
    #         total = sum(p.grad.sum() for p in params)   # 不耗尽
    #         count = sum(1 for p in params)              # 正确计数
    #         print(f"处理了 {count} 个参数")
    #
    #     model = torch.nn.Linear(10, 10)
    #     loss = model(torch.rand(1, 10)).sum()
    #     loss.backward()
    #
    #     print(bad(model.parameters()))
    #     print(good(model.parameters()))

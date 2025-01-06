from typing import Callable

import torch
import torchjd
from torch import Tensor
from torchjd.aggregation import Aggregator


def optimize(
    fn: Callable[[Tensor], Tensor], x0: Tensor, A: Aggregator, lr: float, n_iters: int
) -> tuple[list[Tensor], list[Tensor]]:
    xs = []
    ys = []
    x = x0.clone().requires_grad_()
    optimizer = torch.optim.SGD([x], lr=lr)
    for i in range(n_iters):
        xs.append(x.detach().clone())
        y = fn(x)
        ys.append(y.detach().clone())
        optimizer.zero_grad()
        torchjd.backward(y, aggregator=A)
        optimizer.step()

    return xs, ys

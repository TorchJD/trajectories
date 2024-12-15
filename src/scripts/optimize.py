import pickle
import random
import warnings
from typing import Callable

import numpy as np
import torch
import torchjd
from torch import Tensor
from torchjd.aggregation import (
    IMTLG,
    MGDA,
    Aggregator,
    AlignedMTL,
    CAGrad,
    DualProj,
    GradDrop,
    Mean,
    NashMTL,
    PCGrad,
    Random,
    UPGrad,
)

from trajectories.paths import RESULTS_DIR

warnings.filterwarnings("ignore")


AGGREGATOR_TO_LR = {
    UPGrad(): 0.075,
    MGDA(): 0.075,
    CAGrad(c=0.5): 0.075,
    NashMTL(n_tasks=2): 0.15,
    GradDrop(): 0.0375,
    IMTLG(): 0.15,
    AlignedMTL(): 0.075,
    DualProj(): 0.075,
    PCGrad(): 0.0375,
    Random(): 0.075,
    Mean(): 0.075,
}


def main():
    torch.use_deterministic_algorithms(True)
    RESULTS_DIR.mkdir(exist_ok=True)

    aggregator_to_results = {}

    x0s = [
        torch.tensor([3.0, -2]),
        torch.tensor([0.0, -3.0]),
        torch.tensor([-4.0, 4.0]),
        torch.tensor([-3.0, 4.0]),
        torch.tensor([1.0, 5.0]),
        torch.tensor([-5.0, -1.0]),
    ]

    for A, lr in AGGREGATOR_TO_LR.items():
        print(A)
        aggregator_to_results[str(A)] = []

        for x0 in x0s:
            if isinstance(A, NashMTL):
                A.reset()

            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            xs, ys = optimize(fn1, x0=x0, A=A, lr=lr, n_iters=50)
            aggregator_to_results[str(A)].append((xs, ys))

    with open(RESULTS_DIR / "results.pkl", "wb") as handle:
        pickle.dump(aggregator_to_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
        torchjd.backward(y, [x], A)
        optimizer.step()

    return xs, ys


def fn1(x: Tensor) -> Tensor:
    return x**2


def fn2(x: Tensor) -> Tensor:
    A1 = torch.tensor([[4.0, -4.0], [-4.0, 4.0]])
    u1 = torch.tensor([0.0, 0.0])
    A2 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    u2 = torch.tensor([0.0, 0.0])
    return torch.stack([(x - u1) @ A1 @ (x - u1), (x - u2) @ A2 @ (x - u2)])

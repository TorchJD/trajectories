from typing import Callable

import torch
import torchjd
from torch import Tensor
from torchjd.aggregation import UPGrad, MGDA, CAGrad, NashMTL, GradDrop, IMTLG, AlignedMTL, \
    DualProj, PCGrad, Aggregator, Mean, Random
import warnings
import pickle
warnings.filterwarnings("ignore")

from src.trajectories.paths import RESULTS_DIR

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
    RESULTS_DIR.mkdir(exist_ok=True)

    aggregator_to_results = {}

    x0s = [
        torch.tensor([3., -2]),
        torch.tensor([0., -3.]),
        torch.tensor([-4., 4.]),
        torch.tensor([-3., 4.]),
        torch.tensor([1., 5.]),
        torch.tensor([-5., -1.]),
    ]

    for A, lr in AGGREGATOR_TO_LR.items():
        print(A)
        aggregator_to_results[str(A)] = []

        for x0 in x0s:
            if isinstance(A, NashMTL):
                A.reset()

            xs, ys = optimize(fn1, x0=x0, A=A, lr=lr, n_iters=50)
            aggregator_to_results[str(A)].append((xs, ys))

    with open(RESULTS_DIR / "results.pkl", 'wb') as handle:
        pickle.dump(aggregator_to_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def optimize(
    fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    A: Aggregator,
    lr: float,
    n_iters: int
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
    return x ** 2


def fn2(x: Tensor) -> Tensor:
    A1 = torch.tensor([[4., -4.], [-4., 4.]])
    u1 = torch.tensor([0., 0.])
    A2 = torch.tensor([[0., 0.], [0., 1.]])
    u2 = torch.tensor([0., 0.])
    return torch.stack([(x-u1) @ A1 @ (x-u1), (x-u2) @ A2 @ (x-u2)])


if __name__ == "__main__":
    main()

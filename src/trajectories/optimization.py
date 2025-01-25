import numpy as np
import torch
import torchjd
from torch import Tensor
from torch.nn import functional as F
from torchjd.aggregation import Aggregator

from trajectories.objectives import Objective


def optimize(
    objective: Objective,
    initial_x: Tensor,
    aggregator: Aggregator,
    lr: float,
    n_iters: int,
) -> tuple[list[Tensor], list[Tensor]]:
    xs = []
    ys = []
    x = initial_x.clone().requires_grad_()
    optimizer = torch.optim.SGD([x], lr=lr)
    for i in range(n_iters):
        xs.append(x.detach().clone())
        y = objective(x)
        ys.append(y.detach().clone())
        optimizer.zero_grad()
        torchjd.backward(y, aggregator=aggregator)
        optimizer.step()

    return xs, ys


def compute_gradient_cosine_similarities(
    objective: Objective, x0_min: float, x0_max: float, x1_min: float, x1_max: float, n: int
) -> Tensor:
    if objective.n_values != 2:
        raise ValueError("Objective should have 2 values.")

    x0_len = x0_max - x0_min
    x0_start = x0_min + x0_len / (n * 2)
    x0_end = x0_max - x0_len / (n * 2)

    x1_len = x1_max - x1_min
    x1_start = x1_min + x1_len / (n * 2)
    x1_end = x1_max - x1_len / (n * 2)

    x0s = np.linspace(x0_start, x0_end, n, dtype=np.float32)
    x1s = np.linspace(x1_start, x1_end, n, dtype=np.float32)

    similarities = torch.zeros(n, n)
    for i, x0 in enumerate(x0s):
        for j, x1 in enumerate(x1s):
            x = torch.tensor([x0, x1])
            similarities[i][j] = compute_cosine_similarity(objective, x)

    return similarities


def compute_cosine_similarity(objective: Objective, x: Tensor) -> Tensor:
    J = objective.jacobian(x)
    return F.cosine_similarity(J[0].unsqueeze(0), J[1].unsqueeze(0), eps=1e-19).squeeze()


def compute_objectives_pf_distances(
    pf_points: Tensor, y0_min: float, y0_max: float, y1_min: float, y1_max: float, n: int
) -> Tensor:
    y0_len = y0_max - y0_min
    y0_start = y0_min + y0_len / (n * 2)
    y0_end = y0_max - y0_len / (n * 2)

    y1_len = y1_max - y1_min
    y1_start = y1_min + y1_len / (n * 2)
    y1_end = y1_max - y1_len / (n * 2)

    y0s = np.linspace(y0_start, y0_end, n, dtype=np.float32)
    y1s = np.linspace(y1_start, y1_end, n, dtype=np.float32)

    distances = torch.zeros(n, n)
    for i, y0 in enumerate(y0s):
        for j, y1 in enumerate(y1s):
            y = torch.tensor([y0, y1])
            distances[i][j] = compute_pf_distance(pf_points, y)

    max_distance = torch.max(distances[distances.isfinite()])
    distances = distances / max_distance
    distances[distances.isnan()] = -1.0
    return distances


def compute_pf_distance(pf_points: Tensor, y: Tensor) -> Tensor:
    pf_first = pf_points[:-1, :]
    pf_second = pf_points[1:, :]
    pf_consecutive_diff = pf_first - pf_second
    pf_norms = pf_consecutive_diff.norm(dim=1)
    nominator1 = pf_consecutive_diff[:, 1] * y[0] - pf_consecutive_diff[:, 0] * y[1]
    nominator2 = pf_second[:, 0] * pf_first[:, 1] - pf_second[:, 1] * pf_first[:, 0]
    distances = torch.abs(nominator1 + nominator2) / pf_norms
    return torch.min(distances)

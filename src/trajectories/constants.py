import torch
from torchjd.aggregation import (
    IMTLG,
    MGDA,
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

from trajectories.objectives import ConvexQuadraticForm, ElementWiseQuadratic, QuadraticForm

AGGREGATORS = {
    "upgrad": UPGrad(),
    "mgda": MGDA(),
    "cagrad": CAGrad(c=0.5),
    "nashmtl": NashMTL(n_tasks=2),
    "graddrop": GradDrop(),
    "imtl_g": IMTLG(),
    "aligned_mtl": AlignedMTL(),
    "dualproj": DualProj(),
    "pcgrad": PCGrad(),
    "random": Random(),
    "mean": Mean(),
}
LR_MULTIPLIERS = {
    "upgrad": 1.0,
    "mgda": 1.0,
    "cagrad": 1.0,
    "nashmtl": 2.0,
    "graddrop": 0.5,
    "imtl_g": 2.0,
    "aligned_mtl": 1.0,
    "dualproj": 1.0,
    "pcgrad": 0.5,
    "random": 1.0,
    "mean": 1.0,
}

OBJECTIVES = {
    "EWQ-2": ElementWiseQuadratic(2),
    "QF_v1": QuadraticForm(
        As=[4 * torch.tensor([[1.0, -1.0], [-1.0, 4.0]]), torch.tensor([[1.0, -2.0], [1.0, -1.0]])],
        us=[torch.tensor([1.0, -1.0]), torch.tensor([0.0, 0.0])],
    ),
    "CQF_v1": ConvexQuadraticForm(
        Bs=[
            0.1 * torch.tensor([[1.0, -1.0], [-1.0, 4.0]]),
            torch.tensor([[1.0, -2.0], [1.0, -1.0]]),
        ],
        us=[torch.tensor([1.0, -1.0]), torch.tensor([-1.0, -3.0])],
    ),
    "CQF_v2": ConvexQuadraticForm(
        Bs=[
            4 * torch.tensor([[2.0, 1.0], [1.0, 2.0]]),
            torch.tensor([[2.0, 1.0], [1.0, -2.0]]),
        ],
        us=[torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
    ),
}
BASE_LEARNING_RATES = {
    "EWQ-2": 0.075,
    "QF_v1": 0.01,
    "CQF_v1": 0.1,
    "CQF_v2": 0.002,
}
INITIAL_POINTS = {
    "EWQ-2": [
        [3.0, -2],
        [0.0, -3.0],
        [-4.0, 4.0],
        [-3.0, 4.0],
        [1.0, 5.0],
        [-5.0, -1.0],
    ],
    "QF_v1": [
        [3.0, -2],
        [0.0, -3.0],
        [-4.0, 4.0],
        [-3.0, 4.0],
        [1.0, 5.0],
        [-5.0, -1.0],
    ],
    "CQF_v1": [
        [3.0, -2],
        [0.0, -3.0],
        [-4.0, 4.0],
        [-3.0, 4.0],
        [1.0, 5.0],
        [-5.0, -1.0],
    ],
    "CQF_v2": [
        [0.0, 0.0],
        [-0.5, 0.0],
        [-0.5, 0.5],
        [-0.5, 1.0],
        [0.0, 1.5],
        [0.5, 1.25],
        [1.0, 1.0],
        [1.5, 0.5],
        [1.5, 0.0],
        [1.0, -0.5],
    ],
}
N_ITERS = {
    "EWQ-2": 50,
    "QF_v1": 100,
    "CQF_v1": 200,
    "CQF_v2": 500,
}

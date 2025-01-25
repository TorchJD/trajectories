from math import cos, sin

import numpy as np
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
    "upgrad": UPGrad(reg_eps=1e-7, norm_eps=1e-9),
    "mgda": MGDA(),
    "cagrad": CAGrad(c=0.5),
    "nashmtl": NashMTL(n_tasks=2, optim_niter=1),
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
SUBPLOT_LOCATIONS = {
    "upgrad": (1, 4),
    "mgda": (0, 1),
    "cagrad": (1, 0),
    "nashmtl": (1, 2),
    "graddrop": (0, 3),
    "imtl_g": (0, 4),
    "aligned_mtl": (1, 3),
    "dualproj": (0, 2),
    "random": (1, 1),
    "mean": (0, 0),
    # No location for PCGrad as it's equivalent to UPGrad with 2 objectives
}
LATEX_NAMES = {
    "upgrad": r"$\mathcal A_{\mathrm{UPGrad}}$",
    "mgda": r"$\mathcal A_{\mathrm{MGDA}}$",
    "cagrad": r"$\mathcal A_{\mathrm{CAGrad}}$",
    "nashmtl": r"$\mathcal A_{\mathrm{Nash-MTL}}$",
    "graddrop": r"$\mathcal A_{\mathrm{GradDrop}}$",
    "imtl_g": r"$\mathcal A_{\mathrm{IMTL-G}}$",
    "aligned_mtl": r"$\mathcal A_{\mathrm{Aligned-MTL}}$",
    "dualproj": r"$\mathcal A_{\mathrm{DualProj}}$",
    "pcgrad": r"$\mathcal A_{\mathrm{PCGrad}}$",
    "random": r"$\mathcal A_{\mathrm{RGW}}$",
    "mean": r"$\mathcal A_{\mathrm{Mean}}$",
}

THETA = 0.1
THETA_v5 = np.pi / 16

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
    "CQF_v3": ConvexQuadraticForm(
        Bs=[
            torch.tensor([[cos(THETA), -sin(THETA)], [sin(THETA), cos(THETA)]])
            @ torch.diag(torch.tensor([1.0, 0.0])),
            torch.tensor([[cos(THETA), sin(THETA)], [-sin(THETA), cos(THETA)]])
            @ torch.diag(torch.tensor([2.0, 0.0])),
        ],
        us=[torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0])],
    ),
    "CQF_v4": ConvexQuadraticForm(
        Bs=[
            torch.tensor([[cos(THETA), -sin(THETA)], [sin(THETA), cos(THETA)]])
            @ torch.diag(torch.tensor([1.0, 0.1])),
            torch.tensor([[cos(THETA), sin(THETA)], [-sin(THETA), cos(THETA)]])
            @ torch.diag(torch.tensor([1.5, 0.1])),
        ],
        us=[torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0])],
    ),
    "CQF_v5": ConvexQuadraticForm(
        Bs=[
            torch.tensor([[cos(THETA_v5), -sin(THETA_v5)], [sin(THETA_v5), cos(THETA_v5)]])
            @ torch.diag(torch.tensor([1.0, 0.1])),
            torch.tensor([[cos(THETA_v5), sin(THETA_v5)], [-sin(THETA_v5), cos(THETA_v5)]])
            @ torch.diag(torch.tensor([2.0, 0.1])),
        ],
        us=[torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])],
    ),
}
BASE_LEARNING_RATES = {
    "EWQ-2": 0.075,
    "QF_v1": 0.01,
    "CQF_v1": 0.1,
    "CQF_v2": 0.002,
    "CQF_v3": 0.1,
    "CQF_v4": 0.1,
    "CQF_v5": 0.02,
}
INITIAL_POINTS = {
    "EWQ-2": [
        [3.0, -2.0],
        [0.0, -3.0],
        [-4.0, 4.0],
        [-3.0, 4.0],
        [-3.5, -0.75],
    ],
    "QF_v1": [
        [3.0, -2.0],
        [0.0, -3.0],
        [-4.0, 4.0],
        [-3.0, 4.0],
        [1.0, 5.0],
        [-5.0, -1.0],
    ],
    "CQF_v1": [
        [3.0, -2.0],
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
    "CQF_v3": [
        [0.8, 0.0],
        [0.5, 0.0],
        [0.25, 7.0],
        [0.5, 7.0],
        [0.75, 7.0],
        [0.0, 4.0],
        [1.0, 5.0],
        [0.0625, 0.0],
    ],
    "CQF_v4": [
        [0.8, 0.0],
        [0.5, 0.0],
        [0.25, 7.0],
        [0.5, 7.0],
        [0.75, 7.0],
        [0.0, 4.0],
        [1.0, 5.0],
        [0.0625, 0.0],
    ],
    "CQF_v5": [
        [0.8, 0.0],
        [-1.0, 7.0],
    ],
}
N_ITERS = {
    "EWQ-2": 50,
    "QF_v1": 100,
    "CQF_v1": 200,
    "CQF_v2": 500,
    "CQF_v3": 1000,
    "CQF_v4": 1000,
    "CQF_v5": 2500,
}
N_SAMPLES_SPSM = {
    "EWQ-2": 1,
    "QF_v1": 100,
    "CQF_v1": 100,
    "CQF_v2": 100,
    "CQF_v3": 100,
    "CQF_v4": 100,
    "CQF_v5": 100,
}

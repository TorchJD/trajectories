"""
Optimize the objective using various aggregators. Save the trajectories in the parameter and value
spaces.

Usage:
  optimize <objective>

Arguments:
  <objective         The key of the objective function.

Options:
  -h --help          Show this screen.
"""

import json
import random
import warnings

import numpy as np
import torch
from docopt import docopt
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

from trajectories.objectives import ElementWiseQuadratic, QuadraticForm
from trajectories.optimization import optimize
from trajectories.paths import RESULTS_DIR, get_params_dir, get_values_dir

warnings.filterwarnings("ignore")

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
BASE_LEARNING_RATES = {
    "EWQ-2": 0.075,
    "QF_v1": 0.01,
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
}


def main():
    arguments = docopt(__doc__)
    objective_key = arguments["<objective>"]
    if objective_key not in OBJECTIVES:
        raise ValueError(f"Unknown objective key: {objective_key}")

    aggregator_keys = [
        "upgrad",
        "mgda",
        "cagrad",
        "nashmtl",
        "graddrop",
        "imtl_g",
        "aligned_mtl",
        "dualproj",
        "pcgrad",
        "random",
        "mean",
    ]
    objective = OBJECTIVES[objective_key]
    initial_points = INITIAL_POINTS[objective_key]
    learning_rates = {
        key: BASE_LEARNING_RATES[objective_key] * mult for key, mult in LR_MULTIPLIERS.items()
    }

    torch.use_deterministic_algorithms(True)

    params_dir = get_params_dir(objective_key)
    values_dir = get_values_dir(objective_key)
    params_dir.mkdir(exist_ok=True, parents=True)
    values_dir.mkdir(exist_ok=True, parents=True)

    metadata = {
        "objective_key": objective_key,
        "objective_repr": repr(objective),
        "aggregator_keys": aggregator_keys,
        "aggregator_reprs": {key: repr(AGGREGATORS[key]) for key in aggregator_keys},
        "learning_rates": learning_rates,
        "initial_points": initial_points,
    }
    with open(RESULTS_DIR / objective_key / "metadata.json", "w") as f:
        json.dump(metadata, f)

    results = {}
    for aggregator_key in aggregator_keys:
        aggregator = AGGREGATORS[aggregator_key]
        lr = learning_rates[aggregator_key]
        print(aggregator)
        results[str(aggregator)] = []
        xs_list = []
        ys_list = []
        for initial_point in initial_points:
            if hasattr(aggregator, "reset"):
                aggregator.reset()
            reset_seed()

            x0 = torch.tensor(initial_point)
            xs, ys = optimize(objective, x0=x0, aggregator=aggregator, lr=lr, n_iters=50)

            xs_list.append(torch.stack(xs))
            ys_list.append(torch.stack(ys))

        X = torch.stack(xs_list).numpy()
        Y = torch.stack(ys_list).numpy()
        np.save(params_dir / f"{aggregator_key}.npy", X)
        np.save(values_dir / f"{aggregator_key}.npy", Y)


def reset_seed():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

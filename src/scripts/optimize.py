"""
Optimize the objective using various aggregators. Save the trajectories in the parameter and value
spaces.

Usage:
  optimize <objective> <aggregator>...

Arguments:
  <objective>        The key of the objective function.
  <aggregator>...    The keys of the aggregators to use.

Options:
  -h --help          Show this screen.
"""

import json
import random
import warnings

import numpy as np
import torch
from docopt import docopt

from trajectories.constants import (
    AGGREGATORS,
    BASE_LEARNING_RATES,
    INITIAL_POINTS,
    LR_MULTIPLIERS,
    OBJECTIVES,
)
from trajectories.optimization import optimize
from trajectories.paths import RESULTS_DIR, get_params_dir, get_values_dir

warnings.filterwarnings("ignore")


def main():
    arguments = docopt(__doc__)
    objective_key = arguments["<objective>"]
    if objective_key not in OBJECTIVES:
        raise ValueError(f"Unknown objective key: {objective_key}")

    aggregator_keys = arguments["<aggregator>"]
    for aggregator_key in aggregator_keys:
        if aggregator_key not in AGGREGATORS:
            raise ValueError(f"Unknown aggregator key: {aggregator_key}")

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

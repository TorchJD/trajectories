"""
Plot the trajectories of an objective function in the parameter space.
Usage:
  plot_params <objective>

Arguments:
  <objective         The key of the objective function.

Options:
  -h --help          Show this screen.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt

from trajectories.constants import OBJECTIVES
from trajectories.objectives import ElementWiseQuadratic, Objective
from trajectories.paths import RESULTS_DIR, get_param_plots_dir, get_params_dir
from trajectories.plotters import EWQParamTrajPlotter, ParamTrajPlotter


def main():
    arguments = docopt(__doc__)
    objective_key = arguments["<objective>"]

    # Read metadata.json
    with open(RESULTS_DIR / objective_key / "metadata.json", "r") as f:
        metadata = json.load(f)

    params_dir = get_params_dir(objective_key)
    param_plots_dir = get_param_plots_dir(objective_key)
    param_plots_dir.mkdir(parents=True, exist_ok=True)

    # This seems to be the only way to make the font be Type1, which is the only font type supported
    # by ICML.
    plt.rcParams.update({"text.usetex": True})
    objective = OBJECTIVES[metadata["objective_key"]]
    plotter_cls = objective_to_params_plotter_class(objective)
    aggregator_keys = metadata["aggregator_keys"]
    aggregator_to_X = {key: np.load(params_dir / f"{key}.npy") for key in aggregator_keys}

    for aggregator_key, X in aggregator_to_X.items():
        plotter = plotter_cls(X)
        save_path = param_plots_dir / f"{aggregator_key}.pdf"
        fig, ax = plt.subplots(1, figsize=(2.5, 2.5))
        plotter(ax)
        ax.set_xlim((-6, 6))
        ax.set_ylim((-6, 6))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        plt.savefig(save_path, bbox_inches="tight")


def objective_to_params_plotter_class(objective: Objective) -> type[ParamTrajPlotter]:
    if objective.n_params != 2:
        raise ValueError("Only objectives with 2 parameters are supported.")

    if isinstance(objective, ElementWiseQuadratic):
        return EWQParamTrajPlotter
    else:
        raise NotImplementedError(f"Objective {objective} is not supported.")

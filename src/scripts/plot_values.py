"""
Plot the trajectories of an objective function in the value space.
Usage:
  plot_values <objective>

Arguments:
  <objective>         The key of the objective function.

Options:
  -h --help          Show this screen.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt

from trajectories.constants import OBJECTIVES
from trajectories.objectives import ConvexQuadraticForm, Objective
from trajectories.paths import RESULTS_DIR, get_value_plots_dir, get_values_dir
from trajectories.plotters import CQFValueTrajPlotter, Plotter


def main():
    arguments = docopt(__doc__)
    objective_key = arguments["<objective>"]

    # Read metadata.json
    with open(RESULTS_DIR / objective_key / "metadata.json", "r") as f:
        metadata = json.load(f)

    values_dir = get_values_dir(objective_key)
    value_plots_dir = get_value_plots_dir(objective_key)
    value_plots_dir.mkdir(parents=True, exist_ok=True)

    # This seems to be the only way to make the font be Type1, which is the only font type supported
    # by ICML.
    plt.rcParams.update({"text.usetex": True})
    objective = OBJECTIVES[metadata["objective_key"]]
    aggregator_keys = metadata["aggregator_keys"]
    aggregator_to_Y = {key: np.load(values_dir / f"{key}.npy") for key in aggregator_keys}

    for aggregator_key, Y in aggregator_to_Y.items():
        plotter = build_plotter(objective, Y)
        save_path = value_plots_dir / f"{aggregator_key}.pdf"
        fig, ax = plt.subplots(1, figsize=(2.5, 2.5))
        plotter(ax)
        ax.set_xlim((-0.1, 10))
        ax.set_ylim((-0.1, 10))
        plt.savefig(save_path, bbox_inches="tight")


def build_plotter(objective: Objective, Y: np.ndarray) -> Plotter:
    if objective.n_objectives != 2:
        raise ValueError("Only objectives with 2 dimensions are supported.")

    if isinstance(objective, ConvexQuadraticForm):
        return CQFValueTrajPlotter(objective, Y)
    else:
        raise NotImplementedError(f"Objective {objective} is not supported.")

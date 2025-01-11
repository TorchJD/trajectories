"""
Plot the trajectories of an objective function in the parameter space.
Usage:
  plot <objective>

Arguments:
  <objective         The key of the objective function.

Options:
  -h --help          Show this screen.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt

from trajectories.paths import RESULTS_DIR, get_param_plots_dir, get_params_dir
from trajectories.plotting import plot_parameters


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

    aggregator_keys = metadata["aggregator_keys"]
    aggregator_to_X = {key: np.load(params_dir / f"{key}.npy") for key in aggregator_keys}

    for aggregator_key, X in aggregator_to_X.items():
        plot_parameters(
            X, save_path=param_plots_dir / f"{aggregator_key}.pdf", xlim=(-6, 6), ylim=(-6, 6)
        )

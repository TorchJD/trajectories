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
import torch
from docopt import docopt

from trajectories.constants import N_SAMPLES_SPSM, OBJECTIVES
from trajectories.objectives import WithSPSMappingMixin
from trajectories.paths import RESULTS_DIR, get_value_plots_dir, get_values_dir
from trajectories.plotters import LabelAxesPlotter, MultiTrajPlotter, PFPlotter


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
    objective_key = metadata["objective_key"]
    objective = OBJECTIVES[objective_key]

    if objective.n_values != 2:
        raise ValueError("Can only plot values trajectories for objectives with 2 values.")

    n_samples_spsm = N_SAMPLES_SPSM[objective_key]
    common_plotter = LabelAxesPlotter("Objective $1$", "Objective $2$")

    if isinstance(objective, WithSPSMappingMixin):
        sps_points = objective.sps_mapping.sample(n_samples_spsm, eps=1e-5)
        pf_points = torch.stack([objective(x) for x in sps_points]).numpy()
        common_plotter += PFPlotter(pf_points)

    aggregator_keys = metadata["aggregator_keys"]
    aggregator_to_Y = {key: np.load(values_dir / f"{key}.npy") for key in aggregator_keys}

    for aggregator_key, Y in aggregator_to_Y.items():
        save_path = value_plots_dir / f"{aggregator_key}.pdf"
        fig, ax = plt.subplots(1, figsize=(2.5, 2.5))
        plotter = common_plotter + MultiTrajPlotter(Y)
        plotter(ax)
        plt.savefig(save_path, bbox_inches="tight")

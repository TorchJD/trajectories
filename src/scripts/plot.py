import pickle

import matplotlib.pyplot as plt

from trajectories.paths import RESULTS_DIR
from trajectories.plotting import plot_parameters


def main():
    with open(RESULTS_DIR / "results.pkl", "rb") as handle:
        aggregator_to_results = pickle.load(handle)

    save_dir = RESULTS_DIR / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    # This seems to be the only way to make the font be Type1, which is the only font type supported
    # by ICML.
    plt.rcParams.update({"text.usetex": True})

    for A, results in aggregator_to_results.items():
        plot_parameters(
            results, save_path=save_dir / f"{A}_param_traj.pdf", xlim=(-6, 6), ylim=(-6, 6)
        )

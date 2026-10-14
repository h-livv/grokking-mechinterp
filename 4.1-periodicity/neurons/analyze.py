"""Plot post-ReLU MLP neuron 0 at '=' over all pairs, following Section 4.1.

Run: python 4.1-periodicity/neurons/analyze.py
Neuron 0 is fixed before inspecting activations, matching the paper's example.
"""

import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import collect_grid, draw_map, fourier2d, load_model, paper_style


def main():
    """Save the activation map and its real two-dimensional Fourier spectrum."""
    paper_style()
    model, inputs = load_model()
    _, activation, _ = collect_grid(model, inputs)
    coefficients = fourier2d(activation)
    np.savez_compressed(HERE / "neuron-0.npz", activation=activation, coefficients=coefficients)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    image = draw_map(axes[0], activation, "Neuron 0 at '='")
    fig.colorbar(image, ax=axes[0], label="Post-ReLU activation")
    magnitude = np.abs(coefficients)
    magnitude[0, 0] = 0
    image = draw_map(axes[1], magnitude, "Neuron 0: Fourier spectrum", fourier=True)
    fig.colorbar(image, ax=axes[1], label="Coefficient magnitude")
    fig.tight_layout()
    fig.savefig(HERE / "neuron-0.png", dpi=180)
    plt.close(fig)

    energy = coefficients**2
    energy[0, 0] = 0
    with (HERE / "frequency-energy.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["frequency", "fraction_variation_with_frequency"])
        shares = []
        for k in range(1, (len(energy) + 1) // 2):
            pair = slice(2 * k - 1, 2 * k + 1)
            amount = energy[pair, :].sum() + energy[:, pair].sum() - energy[pair, pair].sum()
            shares.append(amount / energy.sum())
            writer.writerow([k, shares[-1]])
    best = int(np.argmax(shares))
    print(f"Neuron 0: frequency {best + 1}, {100 * shares[best]:.2f}% of variation")


if __name__ == "__main__":
    main()

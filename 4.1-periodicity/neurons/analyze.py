"""Neuron-0 periodicity: calculate the activation spectrum, then plot and save it.

Run: python 4.1-periodicity/neurons/analyze.py
Neuron 0 is fixed in advance; this does not characterize all 512 neurons.
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
from common import collect_grid, draw_map, fourier2d, frequency_shares, load_model, paper_style


def analyze():
    """Return post-ReLU activation [a,b], signed coefficients, and frequency shares.

    A share counts terms containing frequency k in either operand, not only
    terms depending on their sum. No sinusoidal regression is fitted here.
    """
    model, inputs = load_model()
    # Neuron 0 is fixed in advance; capture its post-ReLU value at =.
    _, activation, _ = collect_grid(model, inputs)
    coefficients = fourier2d(activation)
    shares = frequency_shares(coefficients)
    return activation, coefficients, shares


def plot_results(activation, coefficients):
    """Save the activation map beside its Fourier magnitudes, without altering saved values."""
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    image = draw_map(axes[0], activation, "Neuron 0 at '='")
    fig.colorbar(image, ax=axes[0], label="Post-ReLU activation")
    magnitude = np.abs(coefficients)
    magnitude[0, 0] = 0  # Hide the mean only in the plotting copy; saved coefficients retain it.
    image = draw_map(axes[1], magnitude, "Neuron 0: Fourier spectrum", fourier=True)
    fig.colorbar(image, ax=axes[1], label="Coefficient magnitude")
    fig.tight_layout()
    fig.savefig(HERE / "neuron-0.png", dpi=180)
    plt.close(fig)


def save_results(activation, coefficients, shares):
    """Keep the original numerical formats and print neuron 0's dominant frequency."""
    np.savez_compressed(
        HERE / "neuron-0.npz", activation=activation, coefficients=coefficients,
    )
    with (HERE / "frequency-energy.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["frequency", "fraction_variation_with_frequency"])
        writer.writerows(enumerate(shares, start=1))
    best = int(np.argmax(shares))
    print(f"Neuron 0: frequency {best + 1}, {100 * shares[best]:.2f}% of variation")


def main():
    """Analyze neuron 0, draw its maps, and save the same numerical measurements."""
    activation, coefficients, shares = analyze()
    plot_results(activation, coefficients)
    save_results(activation, coefficients, shares)


if __name__ == "__main__":
    main()

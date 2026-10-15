"""Attention periodicity: observe all heads, calculate spectra, then plot/save.

Run: python 4.1-periodicity/attention/analyze.py
Analysis is separate from display; plotting does not select dominant frequencies.
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
    """Return attention maps [a,b,head], real Fourier coefficients, and head summaries.

    Each summary measures energy in cells where either operand has frequency k.
    These are descriptive energy shares, not accuracy or causal-effect scores.
    """
    model, inputs = load_model()
    # Measure softmax weights from = to a, not pre-softmax scores or head outputs.
    patterns, _, _ = collect_grid(model, inputs)
    coefficients = fourier2d(patterns)  # Transform both operand axes while preserving the four heads.
    summaries = []
    for head in range(4):
        shares = frequency_shares(coefficients[:, :, head])
        best = int(np.argmax(shares))
        summaries.append((head, best + 1, shares[best]))
    return patterns, coefficients, summaries


def plot_results(patterns, coefficients):
    """Save both four-head figures; hide the mean only in a display copy."""
    paper_style()
    for filename, fourier in (
        ("attention-patterns.png", False), ("attention-frequencies.png", True),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        for head, ax in enumerate(axes.flat):
            values = np.abs(coefficients[:, :, head]) if fourier else patterns[:, :, head]
            values = values.copy()
            if fourier:
                # Hide the mean only in the plotting copy; saved signed coefficients retain
                # it.
                values[0, 0] = 0
            image = draw_map(ax, values, f"Head {head}: '=' to a", fourier)
            if not fourier:
                image.set_clim(0, 1)
            fig.colorbar(
                image, ax=ax, label="Coefficient magnitude" if fourier else "Attention weight",
            )
        fig.tight_layout()
        fig.savefig(HERE / filename, dpi=180)
        plt.close(fig)


def save_results(patterns, coefficients, summaries):
    """Preserve full signed coefficients, raw maps, and the original CSV format."""
    np.savez_compressed(
        HERE / "attention.npz", patterns=patterns, coefficients=coefficients,
    )
    with (HERE / "summary.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["head", "dominant_frequency", "fraction_variation_with_frequency"])
        writer.writerows(summaries)
    for head, frequency, share in summaries:
        print(f"Head {head}: frequency {frequency}, {100 * share:.2f}% of variation")


def main():
    """Measure the attention structure before producing its plots and data files."""
    patterns, coefficients, summaries = analyze()
    plot_results(patterns, coefficients)
    save_results(patterns, coefficients, summaries)


if __name__ == "__main__":
    main()

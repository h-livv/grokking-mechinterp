"""Plot attention from '=' to a for all four heads, as in Section 4.1.

Run: python 4.1-periodicity/attention/analyze.py
Outputs maps, real Fourier coefficients, and one summary row per head.
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
    """Compare the input maps and their frequency content without selecting heads."""
    paper_style()
    model, inputs = load_model()
    patterns, _, _ = collect_grid(model, inputs)
    coefficients = fourier2d(patterns)
    np.savez_compressed(HERE / "attention.npz", patterns=patterns, coefficients=coefficients)
    for filename, fourier in (("attention-patterns.png", False),
                              ("attention-frequencies.png", True)):
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        for head, ax in enumerate(axes.flat):
            values = np.abs(coefficients[:, :, head]) if fourier else patterns[:, :, head]
            values = values.copy()
            if fourier:
                values[0, 0] = 0  # Hide the input-independent mean in this plot.
            image = draw_map(ax, values, f"Head {head}: '=' to a", fourier)
            if not fourier:
                image.set_clim(0, 1)  # Shared scale for all probability maps.
            fig.colorbar(image, ax=ax, label="Coefficient magnitude" if fourier else "Attention weight")
        fig.tight_layout()
        fig.savefig(HERE / filename, dpi=180)
        plt.close(fig)

    with (HERE / "summary.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["head", "dominant_frequency", "fraction_variation_with_frequency"])
        for head in range(4):
            energy = coefficients[:, :, head]**2
            energy[0, 0] = 0
            # Energy in cells where either operand has this frequency.
            shares = []
            for k in range(1, (len(energy) + 1) // 2):
                pair = slice(2 * k - 1, 2 * k + 1)
                amount = energy[pair, :].sum() + energy[:, pair].sum() - energy[pair, pair].sum()
                shares.append(amount / energy.sum())
            best = int(np.argmax(shares))
            writer.writerow([head, best + 1, shares[best]])
            print(f"Head {head}: frequency {best + 1}, {100 * shares[best]:.2f}% of variation")


if __name__ == "__main__":
    main()

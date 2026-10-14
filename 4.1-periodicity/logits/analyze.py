"""Plot the operand-axis Fourier spectrum of logits, following Section 4.1.

Run: python 4.1-periodicity/logits/analyze.py
The answer axis is pooled by an L2 norm, not Fourier-transformed.
"""

import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import collect_grid, draw_map, fourier2d, load_model, paper_style, SIN_COLOR, COS_COLOR


def main():
    """Save Fourier norms and quantify concentration in equal-frequency blocks."""
    paper_style()
    model, inputs = load_model()
    _, _, logits = collect_grid(model, inputs)
    coefficients = fourier2d(logits)  # [a frequency, b frequency, answer class]
    norms = np.linalg.norm(coefficients, axis=-1)
    np.savez_compressed(HERE / "fourier-norms.npz", norms=norms)
    magnitude = norms.copy()
    magnitude[0, 0] = 0
    fig, ax = plt.subplots(figsize=(7, 6))
    image = draw_map(ax, magnitude, "Logits: operand Fourier spectrum", fourier=True)
    fig.colorbar(image, ax=ax, label="L2 norm over answer classes")
    fig.tight_layout()
    fig.savefig(HERE / "logit-frequencies.png", dpi=180)
    plt.close(fig)

    energy = norms**2
    energy[0, 0] = 0
    # Each frequency pair gives a 2x2 block of cos/cos, cos/sin, sin/cos, sin/sin.
    count = (len(energy) - 1) // 2
    blocks = energy[1:, 1:].reshape(count, 2, count, 2).sum(axis=(1, 3))
    diagonal = np.diag(blocks) / energy.sum()
    # Separate a's cosine/sine contributions; each sums over both b components.
    cos_energy = np.array([energy[2*k-1, 2*k-1:2*k+1].sum() for k in range(1, count+1)]) / energy.sum()
    sin_energy = np.array([energy[2*k, 2*k-1:2*k+1].sum() for k in range(1, count+1)]) / energy.sum()
    np.testing.assert_allclose(cos_energy + sin_energy, diagonal, atol=1e-15)
    with (HERE / "frequency-energy.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["frequency", "fraction_variation_in_equal_frequency_block",
                         "cos_a_fraction", "sin_a_fraction"])
        writer.writerows(zip(range(1, count+1), diagonal, cos_energy, sin_energy))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    frequency = np.arange(1, count + 1)
    ax.bar(frequency - 0.2, 100 * sin_energy, width=0.4, color=SIN_COLOR, label="sin a")
    ax.bar(frequency + 0.2, 100 * cos_energy, width=0.4, color=COS_COLOR, label="cos a")
    ax.set(xlabel="Frequency k", ylabel="Logit variation (%)",
           title="Equal-frequency 2x2 blocks")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "frequency-energy.png", dpi=180)
    plt.close(fig)
    order = np.argsort(-diagonal)[:5]
    # Enlarge the five leading blocks so their four real-basis cells are visible.
    fig, axes = plt.subplots(1, 5, figsize=(12, 3), layout="constrained")
    for ax, k in zip(axes, np.sort(order + 1)):
        pair = slice(2 * k - 1, 2 * k + 1)
        image = ax.imshow(norms[pair, pair].T, origin="upper", cmap="RdBu",
                          norm=TwoSlopeNorm(vmin=-norms.max()/10, vcenter=0, vmax=norms.max()))
        ax.set(title=f"k = {k}", xlabel="a component", ylabel="b component")
        ax.set_xticks([0, 1], ["cos", "sin"])
        ax.set_yticks([0, 1], ["cos", "sin"])
        ax.axvline(0.5, color="white", linewidth=1)
        ax.axhline(0.5, color="white", linewidth=1)
    fig.colorbar(image, ax=list(axes), label="L2 norm", shrink=0.65)
    fig.savefig(HERE / "frequency-blocks.png", dpi=180)
    plt.close(fig)
    largest = np.sort(energy.ravel())[::-1]
    print(f"Top five blocks: {list(order + 1)}, {100 * diagonal[order].sum():.4f}% of variation")
    print(f"Largest 20 real Fourier cells: {100 * largest[:20].sum() / energy.sum():.4f}% of variation")


if __name__ == "__main__":
    main()

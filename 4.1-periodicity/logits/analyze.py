"""Logit periodicity: calculate the operand Fourier structure, then plot/save it.

Run: python 4.1-periodicity/logits/analyze.py
analyze() contains the Fourier and energy calculations. The plotting functions
only display its returned arrays; answer classes are pooled, not transformed.
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


def analyze():
    """Return norms, block-energy shares, and rankings in a named results dictionary.

    logits[a,b,c] -> coefficients[i,j,c] -> norms[i,j]. Only the operand axes
    are transformed. Each equal-frequency block has four real sine/cosine
    cells. Fractions exclude the input-independent cell, but saved norms keep it.
    """
    model, inputs = load_model()
    _, _, logits = collect_grid(model, inputs)
    coefficients = fourier2d(logits)  # Transform a and b, not the answer-class axis c.
    # Pool over answer classes; this loses sign/phase information in the displayed spectrum.
    norms = np.linalg.norm(coefficients, axis=-1)
    energy = norms**2
    energy[0, 0] = 0  # Exclude the input-independent answer-score vector from the energy denominator.
    count = (len(energy) - 1) // 2
    # Group each operand into cos/sin pairs, then sum the four cells of every frequency
    # block.
    blocks = energy[1:, 1:].reshape(count, 2, count, 2).sum(axis=(1, 3))
    # Diagonal blocks use the same frequency for a and b; normalize by all input-dependent
    # energy.
    diagonal = np.diag(blocks) / energy.sum()
    cos_energy, sin_energy = [], []
    for k in range(1, count + 1):
        pair = slice(2 * k - 1, 2 * k + 1)
        # Fix a to cosine while summing both sine/cosine components of b.
        cos_energy.append(energy[2*k-1, pair].sum() / energy.sum())
        # Fix a to sine while summing both sine/cosine components of b.
        sin_energy.append(energy[2*k, pair].sum() / energy.sum())
    cos_energy = np.asarray(cos_energy)
    sin_energy = np.asarray(sin_energy)
    np.testing.assert_allclose(cos_energy + sin_energy, diagonal, atol=1e-15)
    # Rank the five strongest blocks by energy, not by their appearance in the heatmap.
    top_five = np.argsort(-diagonal)[:5]
    largest = np.sort(energy.ravel())[::-1]
    top20_share = largest[:20].sum() / energy.sum()
    return {
        "norms": norms, "frequency": np.arange(1, count + 1),
        "diagonal": diagonal, "cos_energy": cos_energy, "sin_energy": sin_energy,
        "top_five": top_five, "top20_share": top20_share,
    }


def plot_results(result):
    """Save the full spectrum, split energy bars, and enlarged leading blocks.

    The mean is hidden only in the full-spectrum display copy. Enlarged blocks
    use the precomputed ranking and do not select frequencies by appearance.
    """
    paper_style()
    norms = result["norms"]
    magnitude = norms.copy()
    magnitude[0, 0] = 0  # Display-only suppression: the saved norm matrix retains this cell.
    fig, ax = plt.subplots(figsize=(7, 6))
    image = draw_map(ax, magnitude, "Logits: operand Fourier spectrum", fourier=True)
    fig.colorbar(image, ax=ax, label="L2 norm over answer classes")
    fig.tight_layout()
    fig.savefig(HERE / "logit-frequencies.png", dpi=180)
    plt.close(fig)

    frequency = result["frequency"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(
        frequency - 0.2, 100 * result["sin_energy"], width=0.4, color=SIN_COLOR, label="sin a",
    )
    ax.bar(
        frequency + 0.2, 100 * result["cos_energy"], width=0.4, color=COS_COLOR, label="cos a",
    )
    ax.set(
        xlabel="Frequency k", ylabel="Logit variation (%)", title="Equal-frequency 2x2 blocks",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "frequency-energy.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(12, 3), layout="constrained")
    for ax, k in zip(axes, np.sort(result["top_five"] + 1)):
        pair = slice(2 * k - 1, 2 * k + 1)
        image = ax.imshow(
            norms[pair, pair].T, origin="upper", cmap="RdBu",
            norm=TwoSlopeNorm(vmin=-norms.max()/10, vcenter=0, vmax=norms.max()),
        )
        ax.set(title=f"k = {k}", xlabel="a component", ylabel="b component")
        ax.set_xticks([0, 1], ["cos", "sin"])
        ax.set_yticks([0, 1], ["cos", "sin"])
        ax.axvline(0.5, color="white", linewidth=1)
        ax.axhline(0.5, color="white", linewidth=1)
    fig.colorbar(image, ax=list(axes), label="L2 norm", shrink=0.65)
    fig.savefig(HERE / "frequency-blocks.png", dpi=180)
    plt.close(fig)


def save_results(result):
    """Preserve the existing NPZ/CSV formats and print the measured coverage."""
    np.savez_compressed(HERE / "fourier-norms.npz", norms=result["norms"])
    with (HERE / "frequency-energy.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow([
            "frequency", "fraction_variation_in_equal_frequency_block",
            "cos_a_fraction", "sin_a_fraction",
        ])
        writer.writerows(zip(
            result["frequency"], result["diagonal"], result["cos_energy"], result["sin_energy"],
        ))
    order = result["top_five"]
    print(
        f"Top five blocks: {list(order + 1)}, "
        f"{100 * result['diagonal'][order].sum():.4f}% of variation",
    )
    print(f"Largest 20 real Fourier cells: {100 * result['top20_share']:.4f}% of variation")


def main():
    """Calculate the logit structure, plot it, and save the same measurements."""
    result = analyze()
    plot_results(result)
    save_results(result)


if __name__ == "__main__":
    main()

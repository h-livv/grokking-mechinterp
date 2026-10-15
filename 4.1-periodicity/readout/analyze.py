"""Readout periodicity: calculate G's spectrum, then plot and save it.

Run: python 4.1-periodicity/readout/analyze.py
This describes weights; it does not test how activations compute the sum.
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
from common import load_model, fourier_spectrum, paper_style, SIN_COLOR, COS_COLOR


def analyze():
    """Return real Fourier norms, paired energies, and the five strongest frequencies.

    G = U @ W_out maps post-ReLU activations to the MLP contribution to logits.
    It does not include the separate residual-stream contribution.
    """
    model, _ = load_model()
    # Compose [answer,residual] @ [residual,neuron] into [answer,neuron].
    G = (model.unembed.weight.double() @ model.mlp_out.weight.double()).numpy()
    # Transform answer-residue rows, then take norms over the neuron coordinates.
    norms, energy = fourier_spectrum(G)
    # Report the five largest pairs; this fixed summary is not a sparsity threshold.
    top_five = np.argsort(-energy[1:])[:5]
    return norms, energy, top_five


def plot_results(norms):
    """Save the paper-style separate sine/cosine norm plot."""
    paper_style()
    frequency = np.arange(1, (len(norms) + 1) // 2)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(frequency - 0.2, norms[2::2], width=0.4, color=SIN_COLOR, label="sin")
    ax.bar(frequency + 0.2, norms[1::2], width=0.4, color=COS_COLOR, label="cos")
    ax.set(
        title="Fourier components of the neuron-logit map", xlabel="Frequency k",
        ylabel="Norm of Fourier component", xlim=(0, len(frequency) + 1),
    )
    ax.legend()
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(HERE / "fourier-norms.png", dpi=180)
    plt.close(fig)


def save_results(norms, energy, top_five):
    """Keep the existing CSV format and print the measured top-five energy share."""
    with (HERE / "spectrum.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["frequency", "component", "fourier_norm", "component_energy"])
        for index, norm in enumerate(norms):
            component = "constant" if index == 0 else "cos" if index % 2 else "sin"
            writer.writerow([(index+1)//2, component, norm, norm**2])
    print(
        f"Readout: top five frequencies {list(top_five + 1)}, "
        f"{100 * energy[1:][top_five].sum() / energy[1:].sum():.4f}% of nonconstant energy"
    )


def main():
    """Analyze the weight composition, plot the spectrum, and save the measurements."""
    norms, energy, top_five = analyze()
    plot_results(norms)
    save_results(norms, energy, top_five)


if __name__ == "__main__":
    main()

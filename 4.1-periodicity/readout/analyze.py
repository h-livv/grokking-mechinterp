"""Plot the neuron-logit map's sine/cosine norms, as in Figure 3 (right).

Run: python 4.1-periodicity/readout/analyze.py
Only a weight spectrum; Section 4.2's activation-composition tests are separate.
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


def main():
    """Compose the saved output weights and transform their answer-residue axis."""
    paper_style()
    model, _ = load_model()
    G = (model.unembed.weight.double() @ model.mlp_out.weight.double()).numpy()
    norms, energy = fourier_spectrum(G)
    frequency = np.arange(1, (len(norms) + 1) // 2)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(frequency - 0.2, norms[2::2], width=0.4, color=SIN_COLOR, label="sin")
    ax.bar(frequency + 0.2, norms[1::2], width=0.4, color=COS_COLOR, label="cos")
    ax.set(title="Fourier components of the neuron-logit map", xlabel="Frequency k",
           ylabel="Norm of Fourier component", xlim=(0, len(frequency) + 1))
    ax.legend()
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(HERE / "fourier-norms.png", dpi=180)
    plt.close(fig)
    with (HERE / "spectrum.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["frequency", "component", "fourier_norm", "component_energy"])
        for index, norm in enumerate(norms):
            component = "constant" if index == 0 else "cos" if index % 2 else "sin"
            writer.writerow([(index+1)//2, component, norm, norm**2])
    order = np.argsort(-energy[1:])[:5]
    print(f"Readout: top five frequencies {list(order + 1)}, "
          f"{100 * energy[1:][order].sum() / energy[1:].sum():.4f}% of nonconstant energy")


if __name__ == "__main__":
    main()

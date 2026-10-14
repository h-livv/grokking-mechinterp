"""Plot separate sine/cosine embedding norms, following Figure 3 and Section 4.1.

Run: python 4.1-periodicity/embeddings/analyze.py
Reads existing saved weights and writes two plots plus per-component CSV data.
"""

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from common import fourier_spectrum as spectrum, paper_style, SIN_COLOR, COS_COLOR


def main():
    """Compare initial and trained spectra, preserving sine/cosine separation."""
    paper_style()
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    p = manifest["architecture"]["modulus"]
    spectra = {}
    for name in ("initial", "selected"):
        path = ROOT / "results" / f"{name}.pt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files_sha256"][f"results/{name}.pt"]
        weights = torch.load(path, map_location="cpu", weights_only=True)
        # Exclude '='; position embeddings have no residue axis.
        embedding = weights["token_embedding.weight"][:p].double().numpy()
        assert embedding.shape == (p, manifest["architecture"]["width"])
        spectra[name] = spectrum(embedding)

    frequency = np.arange(1, (p + 1) // 2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    titles = ("Initial embeddings (update 0)",
              f"Trained embeddings (update {manifest['selected_step']:,})")
    for ax, name, title in zip(axes, spectra, titles):
        norms, energy = spectra[name]
        ax.bar(frequency - 0.2, norms[2::2], width=0.4, color=SIN_COLOR, label="sin")
        ax.bar(frequency + 0.2, norms[1::2], width=0.4, color=COS_COLOR, label="cos")
        ax.set(title=title, xlabel="Frequency k", ylabel="Norm of Fourier component",
               xlim=(0, (p + 1) // 2))
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.legend()
        share = energy[1:] / energy[1:].sum()
        count = np.searchsorted(np.cumsum(np.sort(share)[::-1]), 0.90) + 1
        print(f"{name}: {count} frequencies explain >=90% nonconstant energy")
    fig.tight_layout()
    fig.savefig(HERE / "fourier-norms.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, label, color in (("initial", "Initial", SIN_COLOR), ("selected", "Trained", COS_COLOR)):
        energy = spectra[name][1][1:]
        ax.plot(frequency, 100 * np.cumsum(np.sort(energy)[::-1]) / energy.sum(),
                label=label, color=color)
    ax.axhline(90, color="gray", linestyle="--", linewidth=1, label="90% energy")
    ax.set(xlabel="Number of frequencies, ranked by energy",
           ylabel="Cumulative nonconstant energy (%)", xlim=(1, len(frequency)), ylim=(0, 102))
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(HERE / "energy-concentration.png", dpi=180)
    plt.close(fig)

    with (HERE / "spectrum.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["checkpoint", "frequency", "component", "fourier_norm",
                         "component_energy", "fraction_total_energy", "fraction_nonconstant_energy"])
        for name, (norms, energy) in spectra.items():
            for index, norm in enumerate(norms):
                k = (index + 1) // 2
                component = "constant" if index == 0 else "cos" if index % 2 else "sin"
                writer.writerow([name, k, component, norm, norm**2, norm**2 / energy.sum(),
                                 norm**2 / energy[1:].sum() if index else ""])


if __name__ == "__main__":
    main()

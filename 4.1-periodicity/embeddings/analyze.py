"""Embedding periodicity: calculate first, then plot and save the results.

Run: python 4.1-periodicity/embeddings/analyze.py
analyze() contains the Fourier measurements; plot_results() only draws them.
save_results() writes the component table. No model forward pass is needed.
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


def analyze():
    """Measure initial/trained embedding spectra and their energy concentration.

    E[residue,coordinate] excludes '='. norms separates cosine/sine components;
    energy combines their squared norms. cumulative/count90 summarize sparsity.
    Return these measurements and the selected checkpoint's training update.
    """
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    p = manifest["architecture"]["modulus"]
    spectra = {}
    for name in ("initial", "selected"):
        path = ROOT / "results" / f"{name}.pt"
        expected = manifest["files_sha256"][f"results/{name}.pt"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        weights = torch.load(path, map_location="cpu", weights_only=True)
        # Exclude the = token; position embeddings have no residue axis.
        embedding = weights["token_embedding.weight"][:p].double().numpy()
        assert embedding.shape == (p, manifest["architecture"]["width"])
        norms, energy = spectrum(embedding)
        # Normalize by variation across residues, excluding the constant component.
        share = energy[1:] / energy[1:].sum()
        cumulative = np.cumsum(np.sort(energy[1:])[::-1]) / energy[1:].sum()
        # Smallest number of ranked frequency pairs covering at least 90% of variation.
        count90 = np.searchsorted(np.cumsum(np.sort(share)[::-1]), 0.90) + 1
        spectra[name] = {
            "norms": norms, "energy": energy,
            "cumulative": cumulative, "count90": count90,
        }
    return spectra, manifest["selected_step"]


def plot_results(spectra, selected_step):
    """Save separate sine/cosine norms and cumulative energy; do not fit or transform data."""
    paper_style()
    count = len(spectra["initial"]["energy"]) - 1
    frequency = np.arange(1, count + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    titles = ("Initial embeddings (update 0)", f"Trained embeddings (update {selected_step:,})")
    for ax, name, title in zip(axes, spectra, titles):
        norms = spectra[name]["norms"]
        ax.bar(frequency - 0.2, norms[2::2], width=0.4, color=SIN_COLOR, label="sin")
        ax.bar(frequency + 0.2, norms[1::2], width=0.4, color=COS_COLOR, label="cos")
        ax.set(
            title=title, xlabel="Frequency k", ylabel="Norm of Fourier component",
            xlim=(0, count + 1),
        )
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "fourier-norms.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, label, color in (("initial", "Initial", SIN_COLOR), ("selected", "Trained", COS_COLOR)):
        ax.plot(
            frequency, 100 * spectra[name]["cumulative"], label=label, color=color,
        )
    ax.axhline(90, color="gray", linestyle="--", linewidth=1, label="90% energy")
    ax.set(
        xlabel="Number of frequencies, ranked by energy", ylabel="Cumulative nonconstant energy (%)",
        xlim=(1, count), ylim=(0, 102),
    )
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(HERE / "energy-concentration.png", dpi=180)
    plt.close(fig)


def save_results(spectra):
    """Write the existing per-component CSV format and print the 90% coverage counts."""
    with (HERE / "spectrum.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow([
            "checkpoint", "frequency", "component", "fourier_norm",
            "component_energy", "fraction_total_energy", "fraction_nonconstant_energy",
        ])
        for name, result in spectra.items():
            norms, energy = result["norms"], result["energy"]
            for index, norm in enumerate(norms):
                k = (index + 1) // 2
                component = "constant" if index == 0 else "cos" if index % 2 else "sin"
                writer.writerow([
                    name, k, component, norm, norm**2, norm**2 / energy.sum(),
                    norm**2 / energy[1:].sum() if index else "",
                ])
            print(f"{name}: {result['count90']} frequencies explain >=90% nonconstant energy")


def main():
    """Calculate the spectra, draw their figures, then save their numerical values."""
    spectra, selected_step = analyze()
    plot_results(spectra, selected_step)
    save_results(spectra)


if __name__ == "__main__":
    main()

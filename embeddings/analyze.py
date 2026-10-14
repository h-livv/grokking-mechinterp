"""Plot embedding periodicity following Section 4.1 of Nanda et al.

Run with: python embeddings/analyze.py
Uses the existing public initial/selected weights; no training or predictions.
Outputs two plots and a CSV beside this script.
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def spectrum(embedding):
    """Return Fourier norms and energies for a real [residue, feature] table.

    Transform the residue axis, then take the L2 norm over embedding features.
    Orthonormal normalization makes squared norms add to the table's energy.
    Positive and negative frequencies are conjugate pairs for real embeddings.
    """
    p = embedding.shape[0]
    assert p >= 3 and p % 2 == 1
    assert np.isfinite(embedding).all()
    fourier = np.fft.fft(embedding, axis=0, norm="ortho")
    np.testing.assert_allclose(
        np.fft.ifft(fourier, axis=0, norm="ortho"), embedding, atol=1e-12
    )
    np.testing.assert_allclose(
        fourier[1:], fourier[:0:-1].conj(), atol=1e-12
    )
    norms = np.linalg.norm(fourier[: (p + 1) // 2], axis=1)
    energy = norms**2
    energy[1:] *= 2  # Account for the omitted negative-frequency partners.
    np.testing.assert_allclose(energy.sum(), np.sum(embedding**2), rtol=1e-12)
    return norms, energy


def main():
    """Load saved embeddings, compare spectra, and save reproducible outputs."""
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    p = manifest["architecture"]["modulus"]
    spectra = {}
    for name in ("initial", "selected"):
        path = ROOT / "results" / f"{name}.pt"
        expected = manifest["files_sha256"][f"results/{name}.pt"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        weights = torch.load(path, map_location="cpu", weights_only=True)
        # The final row is '='. Positions are separate parameters, not residues.
        embedding = weights["token_embedding.weight"][:p].double().numpy()
        assert embedding.shape == (p, manifest["architecture"]["width"])
        spectra[name] = spectrum(embedding)

    # Paper-style norms: only k=1,...,56; report the constant k=0 separately.
    frequency = np.arange(1, (p + 1) // 2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    titles = ("Initial embeddings (update 0)",
              f"Trained embeddings (update {manifest['selected_step']:,})")
    for ax, name, title in zip(axes, spectra, titles):
        norms, energy = spectra[name]
        ax.bar(frequency, norms[1:], color="#2463a5")
        ax.set(title=title, xlabel="Frequency k", ylabel="Fourier L2 norm",
               xlim=(0, (p + 1) // 2))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        share = energy[1:] / energy[1:].sum()
        order = np.argsort(-share)
        count = np.searchsorted(np.cumsum(share[order]), 0.90) + 1
        print(f"{name}: {count} frequencies explain >=90% nonconstant energy; "
              f"Constant component = {100 * energy[0] / energy.sum():.6f}% of total energy")
        if name == "selected":
            for index in order[:count]:
                ax.annotate(str(index + 1), (index + 1, norms[index + 1]),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", rotation=90, fontsize=9)
            ax.set_ylim(0, norms[1:].max() * 1.15)
    fig.tight_layout()
    fig.savefig(HERE / "fourier-norms.png", dpi=180)
    plt.close(fig)

    # Normalized concentration can be compared despite changing weight scales.
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, label, color in (("initial", "Initial", "#2463a5"),
                               ("selected", "Trained", "#c24b38")):
        energy = spectra[name][1][1:]
        cumulative = np.cumsum(np.sort(energy)[::-1]) / energy.sum()
        ax.plot(frequency, 100 * cumulative, label=label, color=color)
    ax.axhline(90, color="gray", linestyle="--", linewidth=1, label="90% energy")
    ax.set(xlabel="Number of frequencies, ranked by energy",
           ylabel="Cumulative nonconstant energy (%)", xlim=(1, len(frequency)),
           ylim=(0, 102))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(HERE / "energy-concentration.png", dpi=180)
    plt.close(fig)

    with (HERE / "spectrum.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["checkpoint", "frequency", "fourier_norm",
                         "paired_energy", "fraction_total_energy",
                         "fraction_non_dc_energy"])
        for name, (norms, energy) in spectra.items():
            for k in range(len(norms)):
                writer.writerow([name, k, norms[k], energy[k],
                                 energy[k] / energy.sum(),
                                 energy[k] / energy[1:].sum() if k else ""])


if __name__ == "__main__":
    main()

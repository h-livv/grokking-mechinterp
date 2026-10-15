"""Section 4.2 weight composition: calculate low-rank/Fourier fidelity, then plot/save.

Run: python 4.2-weights/readout/analyze.py
analyze() measures G; plot_results() draws fixed measurements; save_results()
records the selected frequencies, matrices, singular values, and errors.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import ROOT, load_model, readout_components, paper_style, SIN_COLOR, COS_COLOR


def analyze():
    """Compare the selected sine/cosine reconstruction with the optimal same-rank SVD."""
    model, _ = load_model()
    result = readout_components(model)
    G, approximation = result["G"], result["selected"]
    singular = np.linalg.svd(G, compute_uv=False)
    rank = len(result["indices"])
    cumulative = np.cumsum(singular**2) / np.sum(singular**2)
    tolerance = singular[0] * max(G.shape) * np.finfo(float).eps
    error = np.linalg.norm(G - approximation) / np.linalg.norm(G)
    optimal_error = np.sqrt(np.sum(singular[rank:]**2) / np.sum(singular**2))
    # A high-energy rank-10 approximation does not imply that G has exact rank 10.
    metrics = {"selected_real_dimensions": rank,
               "numerical_rank": int(np.sum(singular > tolerance)),
               "numerical_rank_tolerance": float(tolerance),
               "selected_frequency_energy_fraction": float(result["energy"][result["frequencies"]-1].sum()
                                                            / result["energy"].sum()),
               "fourier_relative_frobenius_error": float(error),
               "svd_relative_frobenius_error": float(optimal_error),
               "svd_retained_energy_fraction": float(cumulative[rank-1])}
    result.update(singular=singular, cumulative=cumulative, metrics=metrics)
    return result


def plot_results(result):
    """Plot sine/cosine readout norms, singular values, and reconstruction errors."""
    paper_style()
    norms = np.linalg.norm(result["coefficients"], axis=1)
    frequency = np.arange(1, len(result["energy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(frequency-0.2, norms[2::2], width=0.4, color=SIN_COLOR, label="sin")
    axes[0].bar(frequency+0.2, norms[1::2], width=0.4, color=COS_COLOR, label="cos")
    axes[0].set(title="Fourier components of G = U W_out", xlabel="Frequency k", ylabel="Component norm")
    axes[0].legend()
    singular = result["singular"]
    axes[1].semilogy(np.arange(1, len(singular)+1), singular / singular[0], color=SIN_COLOR)
    axes[1].axvline(len(result["indices"]), color="gray", linestyle="--")
    axes[1].set(title="Singular values of G", xlabel="Singular value index", ylabel="Value / largest value")
    for ax in axes:
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(HERE / "readout-composition.png", dpi=180)
    plt.close(fig)

    metrics = result["metrics"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Selected Fourier directions", "Optimal same-rank SVD"],
           100*np.array([metrics["fourier_relative_frobenius_error"], metrics["svd_relative_frobenius_error"]]),
           color=[COS_COLOR, SIN_COLOR])
    ax.set(ylabel="Relative Frobenius error (%)", title="Readout reconstruction")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(HERE / "reconstruction-error.png", dpi=180)
    plt.close(fig)


def save_results(result):
    """Save the weight evidence and freeze the local selection rule in frequencies.json."""
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    selection = {"checkpoint": "results/selected.pt", "selected_step": manifest["selected_step"],
                 "checkpoint_sha256": hashlib.sha256((ROOT / "results/selected.pt").read_bytes()).hexdigest(),
                 "rule": "smallest paired-frequency set covering >=90% nonconstant G energy; cap 8",
                 "frequencies": result["frequencies"].tolist(),
                 "development_fit_seed": 0, "discovery_pairs": 2234, "fit_validation_pairs": 2235}
    (HERE.parent / "frequencies.json").write_text(json.dumps(selection, indent=2) + "\n")
    (HERE / "metrics.json").write_text(json.dumps(result["metrics"], indent=2) + "\n")
    np.savez_compressed(HERE / "readout.npz", G=result["G"], G_selected=result["selected"],
                        directions=result["directions"], singular_values=result["singular"],
                        frequencies=result["frequencies"])
    norms = np.linalg.norm(result["coefficients"], axis=1)
    with (HERE / "spectrum.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["frequency", "cos_norm", "sin_norm", "nonconstant_energy_fraction", "selected"])
        for k, energy in enumerate(result["energy"], start=1):
            writer.writerow([k, norms[2*k-1], norms[2*k], energy/result["energy"].sum(),
                             k in result["frequencies"]])
    with (HERE / "singular-values.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["index", "singular_value", "cumulative_energy_fraction"])
        writer.writerows(zip(range(1, len(result["singular"])+1), result["singular"], result["cumulative"]))
    print("Readout:", result["frequencies"].tolist(), result["metrics"])


def main():
    """Calculate weight composition, plot it, and save its reproducible evidence."""
    result = analyze()
    plot_results(result)
    save_results(result)


if __name__ == "__main__":
    main()

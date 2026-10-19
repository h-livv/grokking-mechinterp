"""Section 4.3: fit individual neurons and test frequency localization of their readout.

Run: python 4.3-neurons/neurons/analyze.py
Mathematics lives in analyze(); plotting and saving are separate below.
"""

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fitting import (TERMS, load_model, fixed_readout, collect_activations, fit_partitions,
                     fit_quadratics, predict_quadratics, paper_style, SIN_COLOR, COS_COLOR)


def analyze():
    """Fit all 512 post-ReLU neurons and cluster those with discovery R² > 0.85.

    Each neuron chooses one of the fixed key frequencies on discovery pairs.
    Readout localization is measured independently from G's output Fourier rows.
    """
    model, tokens = load_model()
    weights = fixed_readout(model)
    hidden, _, _ = collect_activations(model, tokens)
    inputs, frequencies = tokens.numpy(), weights["frequencies"]
    partitions = fit_partitions()
    result = fit_quadratics(inputs, hidden, frequencies, partitions)
    columns = np.arange(hidden.shape[1])
    discovery = result["discovery_r2"][result["best"], columns]
    validation = result["validation_r2"][result["best"], columns]
    eligible = discovery > 0.85
    paired = (weights["coefficients"][1:].reshape(56, 2, -1)**2).sum(axis=1)
    assigned = frequencies[result["best"]]
    own_energy = paired[assigned-1, columns]
    share = own_energy / paired.sum(axis=0)
    output_best = paired.argmax(axis=0)+1
    rows = [{"neuron": int(j), "frequency": int(assigned[j]), "eligible": bool(eligible[j]),
             "r2_discovery": float(discovery[j]), "r2_validation": float(validation[j]),
             "readout_frequency": int(output_best[j]), "matching_readout_energy_fraction": float(share[j])}
            for j in columns]
    cluster_rows = []
    for k in frequencies:
        selected = eligible & (assigned == k)
        total = paired[:, selected].sum()
        cluster_rows.append({"frequency": int(k), "count": int(selected.sum()),
                             "matching_output_frequency_count": int(np.sum(output_best[selected] == k)),
                             "matching_readout_energy_fraction": float(paired[k-1, selected].sum()/total) if total else None})
    metrics = {"neurons": len(columns), "discovery_r2_above_0_85": int(eligible.sum()),
               "validation_r2_above_0_85": int(np.sum(validation > 0.85)),
               "eligible_and_validation_above_0_85": int(np.sum(eligible & (validation > 0.85))),
               "eligible_matching_output_frequency": int(np.sum(eligible & (output_best == assigned))),
               "threshold": 0.85, "selection_split": "discovery"}
    result.update(inputs=inputs, hidden=hidden, predicted=predict_quadratics(inputs, frequencies, result["best"], result["coefficients"]),
                  frequencies=frequencies, eligible=eligible, rows=rows, clusters=cluster_rows, metrics=metrics,
                  fourier_readout=weights["coefficients"], fit_ids=partitions["discovery"], validation_ids=partitions["validation"])
    return result


def plot_results(result):
    """Plot all-neuron fit quality, each readout cluster, and a fixed neuron-0 example."""
    paper_style()
    columns = np.arange(len(result["best"]))
    discovery = result["discovery_r2"][result["best"], columns]
    validation = result["validation_r2"][result["best"], columns]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(min(0, validation.min()), 1, 26)
    axes[0].hist([discovery, validation], bins=bins, color=[COS_COLOR, SIN_COLOR], label=["Discovery", "Fit validation"])
    axes[0].axvline(0.85, linestyle="--", color="gray", label="85% threshold")
    axes[0].set(xlabel="Single-frequency polynomial R²", ylabel="Neuron count", title="All 512 neurons")
    axes[0].legend(fontsize=8)
    axes[1].bar([str(row["frequency"]) for row in result["clusters"]], [row["count"] for row in result["clusters"]], color=SIN_COLOR)
    axes[1].set(xlabel="Assigned frequency", ylabel="Neuron count", title="Clusters selected on discovery")
    fig.tight_layout()
    fig.savefig(HERE / "fit-quality.png", dpi=180)
    plt.close(fig)

    for index, k in enumerate(result["frequencies"]):
        ids = np.flatnonzero(result["eligible"] & (result["best"] == index))
        if not len(ids):
            continue
        values = result["fourier_readout"][:, ids]
        bound = abs(values[1:]).max()
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
        for ax, offset, name in zip(axes, [1, 2], ["cos", "sin"]):
            image = ax.imshow(values[offset::2], aspect="auto", cmap="RdBu", vmin=-bound, vmax=bound, origin="lower")
            ax.set(title=f"{name} output components: k={k} cluster", xlabel="Neuron (original index)", ylabel="Output frequency")
            ticks = np.unique(np.linspace(0, len(ids)-1, min(6, len(ids)), dtype=int))
            ax.set_xticks(ticks, ids[ticks])
            ticks = sorted(set([0, 13, 27, 41, 55, int(k)-1]))
            ax.set_yticks(ticks, np.array(ticks)+1)
        fig.colorbar(image, ax=axes, label="Readout Fourier coefficient", shrink=0.8)
        fig.savefig(HERE / f"readout-frequency-{k}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    observed, predicted = result["hidden"][:, 0], result["predicted"][:, 0]
    low, high = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
    for ax, values, title in zip(axes[:2], [observed, predicted], ["Observed neuron 0", "Single-frequency polynomial"]):
        image = ax.imshow(values.reshape(113, 113).T, cmap="Blues", vmin=low, vmax=high, origin="upper")
        ax.set(title=title, xlabel="Operand a", ylabel="Operand b")
        fig.colorbar(image, ax=ax, label="Activation")
    ids = result["validation_ids"]
    axes[2].scatter(observed[ids], predicted[ids], s=4, alpha=0.2, color=SIN_COLOR)
    axes[2].plot([low, high], [low, high], "--", color="gray")
    axes[2].set(title=f"k={result['rows'][0]['frequency']}; validation R²={result['rows'][0]['r2_validation']:.3f}",
                xlabel="Observed activation", ylabel="Predicted activation")
    fig.tight_layout()
    fig.savefig(HERE / "neuron-0.png", dpi=180)
    plt.close(fig)


def save_results(result):
    """Save every neuron's fit, cluster summaries, and coefficients needed by Section 4.4."""
    for name, rows in [("fits.csv", result["rows"]), ("clusters.csv", result["clusters"])]:
        with (HERE / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(HERE / "fits.npz", coefficients=result["coefficients"], best=result["best"],
                        frequencies=result["frequencies"], eligible=result["eligible"], terms=np.array(TERMS),
                        discovery_r2=result["discovery_r2"], validation_r2=result["validation_r2"],
                        fit_ids=result["fit_ids"], validation_ids=result["validation_ids"])
    (HERE / "metrics.json").write_text(json.dumps(result["metrics"], indent=2, allow_nan=False)+"\n")
    print("Neurons:", result["metrics"])
    for row in result["clusters"]:
        print(row)


def main():
    """Analyze, then plot and save the neuron fits."""
    result = analyze()
    plot_results(result)
    save_results(result)


if __name__ == "__main__":
    main()

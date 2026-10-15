"""Section 4.2 logit approximation: calculate fidelity/loss, then plot and save.

Run: python 4.2-weights/logits/analyze.py
Fit one cos(k(a+b-c)) coefficient per selected frequency on development
discovery pairs. Keep the residual route when checking the exact decomposition.
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
from shared import (load_model, readout_components, collect_activations, fit_partitions,
                    cosine_scores, r_squared, score_logits, paper_style, SIN_COLOR, COS_COLOR)


def analyze():
    """Fit the paper's cosine score family and compare it with actual model logits.

    Regression is zero-intercept OLS against raw logits. Accuracy/NLL are
    evaluated only on train/development IDs. Full-grid R² is descriptive.
    """
    model, inputs = load_model()
    weights = readout_components(model)
    hidden, x1, actual = collect_activations(model, inputs)
    inputs = inputs.numpy()
    partitions = fit_partitions()
    fit, validation = partitions["discovery"], partitions["validation"]
    mlp = hidden @ weights["G"].T
    residual = x1 @ weights["U"].T
    reconstructed = residual + mlp
    error = np.abs(actual - reconstructed)
    # The identity is exact algebraically; float32 forward rounding remains in the captured tensors.
    np.testing.assert_allclose(reconstructed, actual, atol=5e-5, rtol=1e-6)
    features = cosine_scores(inputs, weights["frequencies"])
    # Each discovery input contributes all answer scores; columns are the selected frequencies.
    coefficients, _, rank, _ = np.linalg.lstsq(features[fit].reshape(-1, len(weights["frequencies"])),
                                              actual[fit].ravel(), rcond=None)
    assert rank == len(coefficients)
    predicted = np.einsum("nck,k->nc", features, coefficients)
    conditions = {"intact": actual, "reconstructed": reconstructed, "mlp_only": mlp,
                  "residual_only": residual, "selected_readout_plus_residual": hidden @ weights["selected"].T + residual,
                  "cosine_approximation": predicted}
    rows = []
    for split, ids in partitions.items():
        centered_actual = actual[ids] - actual[ids].mean(axis=1, keepdims=True)
        for condition, logits in conditions.items():
            accuracy, nll = score_logits(logits, inputs, ids)
            centered = logits[ids] - logits[ids].mean(axis=1, keepdims=True)
            rows.append({"split": split, "condition": condition, "count": len(ids),
                         "accuracy": accuracy, "nll": nll, "raw_logit_r2": r_squared(actual[ids], logits[ids]),
                         "centered_logit_r2": r_squared(centered_actual, centered)})
    metrics = {"reconstruction_max_absolute_error": float(error.max()),
               "reconstruction_relative_frobenius_error": float(np.linalg.norm(actual-reconstructed)/np.linalg.norm(actual)),
               "full_grid_raw_logit_r2": r_squared(actual, predicted),
               "validation_raw_logit_r2": r_squared(actual[validation], predicted[validation]),
               "final_accuracy_or_loss_evaluated": False}
    return {"inputs": inputs, "actual": actual, "predicted": predicted,
            "frequencies": weights["frequencies"], "coefficients": coefficients,
            "rows": rows, "metrics": metrics, "fit_ids": fit, "validation_ids": validation}


def plot_results(result):
    """Plot frozen coefficients, all fit-validation logit scores, and validation loss."""
    paper_style()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(np.arange(len(result["frequencies"])), result["coefficients"], color=COS_COLOR)
    ax.set_xticks(np.arange(len(result["frequencies"])), result["frequencies"])
    ax.set(xlabel="Frequency k", ylabel="Fitted coefficient", title="Cosine score approximation")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(HERE / "coefficients.png", dpi=180)
    plt.close(fig)

    ids = result["validation_ids"]
    actual, predicted = result["actual"][ids], result["predicted"][ids]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.hexbin(actual.ravel(), predicted.ravel(), gridsize=70, mincnt=1, bins="log", cmap="Blues")
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], color="gray", linestyle="--")
    ax.set(xlabel="Actual model logit", ylabel="Cosine approximation", title=f"Fit-validation R² = {result['metrics']['validation_raw_logit_r2']:.4f}")
    fig.colorbar(image, ax=ax, label="Score count (log scale)")
    fig.tight_layout()
    fig.savefig(HERE / "logit-fit.png", dpi=180)
    plt.close(fig)

    rows = [row for row in result["rows"] if row["split"] == "validation" and
            row["condition"] in ("intact", "cosine_approximation")]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Intact model", "Cosine approximation"], [row["nll"] for row in rows], color=[SIN_COLOR, COS_COLOR])
    ax.set(yscale="log", ylabel="Fit-validation NLL (nats)", title="Prediction loss")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(HERE / "loss-comparison.png", dpi=180)
    plt.close(fig)


def save_results(result):
    """Save fit coefficients, partition IDs, predictive metrics, and reconstruction checks."""
    with (HERE / "coefficients.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["frequency", "coefficient"])
        writer.writerows(zip(result["frequencies"], result["coefficients"]))
    with (HERE / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result["rows"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["rows"])
    (HERE / "metrics.json").write_text(json.dumps(result["metrics"], indent=2) + "\n")
    np.savez_compressed(HERE / "fit.npz", coefficients=result["coefficients"], frequencies=result["frequencies"],
                        fit_ids=result["fit_ids"], validation_ids=result["validation_ids"])
    print("Logit reconstruction and approximation:", result["metrics"])
    for row in result["rows"]:
        if row["split"] == "validation":
            print(row)


def main():
    """Calculate approximation fidelity before plotting and saving its evidence."""
    result = analyze()
    plot_results(result)
    save_results(result)


if __name__ == "__main__":
    main()

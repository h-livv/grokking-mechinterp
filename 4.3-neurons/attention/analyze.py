"""Section 4.3's attention-head evidence, including the Appendix C.1.2 comparison.

Run: python 4.3-neurons/attention/analyze.py
Analyze actual head outputs, attention patterns, and QK/OV weight composition.
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
from fitting import (load_model, fixed_readout, collect_heads, fit_partitions, quadratic_features,
                     r_squared, paper_style, SIN_COLOR, COS_COLOR)


def vector_r2(actual, predicted):
    """Pool squared errors over coordinates, centring each output coordinate separately."""
    return float(1-np.sum((actual-predicted)**2)/np.sum((actual-actual.mean(axis=0))**2))


def analyze():
    """Compare single-frequency quadratic and all-key linear fits on frozen head outputs.

    A head's one frequency is chosen by pooled discovery R², not independently
    for each coordinate. QK and OV spectra come directly from composed weights.
    """
    model, tokens = load_model()
    weights = fixed_readout(model)
    inputs, frequencies = tokens.numpy(), weights["frequencies"]
    heads, attention = collect_heads(model, tokens)
    partitions = fit_partitions()
    fit, validation = partitions["discovery"], partitions["validation"]
    features = [quadratic_features(inputs, k) for k in frequencies]
    linear = np.column_stack([np.ones(len(inputs))]+[X[:, j] for X in features for j in range(1, 5)])
    assert np.linalg.matrix_rank(linear[fit]) == linear.shape[1]
    rows, polynomial, linear_coefficients, patterns = [], [], [], []
    best, predicted = [], np.empty_like(heads)
    E = model.token_embedding.weight[:113].double().numpy()
    equal = (model.token_embedding.weight[113]+model.position_embedding[2]).double().numpy()
    qk, ov = [], []
    for head in range(4):
        target = heads[:, head]
        candidates = [np.linalg.lstsq(X[fit], target[fit], rcond=None)[0] for X in features]
        scores = [vector_r2(target[fit], X[fit] @ C) for X, C in zip(features, candidates)]
        index = int(np.argmax(scores))
        k, X, coefficient = int(frequencies[index]), features[index], candidates[index]
        best.append(index)
        polynomial.append(coefficient)
        predicted[:, head] = X @ coefficient
        linear_coefficient = np.linalg.lstsq(linear[fit], target[fit], rcond=None)[0]
        linear_coefficients.append(linear_coefficient)
        # The paper's pattern formula fixes the intercept at 1/2. Self-attention is measured below.
        pattern_candidates = []
        for candidate in features:
            D = np.column_stack([candidate[:, 1]-candidate[:, 3], candidate[:, 2]-candidate[:, 4]])
            c = np.linalg.lstsq(D[fit], attention[fit, head, 0]-0.5, rcond=None)[0]
            pattern_candidates.append((r_squared(attention[fit, head, 0], 0.5+D[fit] @ c), c, D))
        pattern_index = int(np.argmax([item[0] for item in pattern_candidates]))
        pattern_score, pattern_coefficient, D = pattern_candidates[pattern_index]
        patterns.append(pattern_coefficient)
        block = slice(32*head, 32*(head+1))
        Q, K = model.query.weight.double().numpy()[block], model.key.weight.double().numpy()[block]
        V, O = model.value.weight.double().numpy()[block], model.attention_out.weight.double().numpy()[:, block]
        qk.append(weights["basis"].T @ (E @ K.T @ Q @ equal / np.sqrt(32)))
        ov.append(weights["basis"].T @ (E @ V.T @ O.T))
        rows.append({"head": head, "quadratic_frequency": k, "quadratic_r2_discovery": scores[index],
                     "quadratic_r2_validation": vector_r2(target[validation], predicted[validation, head]),
                     "all_key_linear_r2_validation": vector_r2(target[validation], linear[validation] @ linear_coefficient),
                     "pattern_frequency": int(frequencies[pattern_index]), "pattern_alpha": float(pattern_coefficient[0]),
                     "pattern_beta": float(pattern_coefficient[1]), "pattern_r2_discovery": pattern_score,
                     "pattern_r2_validation": r_squared(attention[validation, head, 0], 0.5+D[validation] @ pattern_coefficient),
                     "mean_self_attention": float(attention[:, head, 2].mean()),
                     "max_self_attention": float(attention[:, head, 2].max())})
    qk, ov = np.array(qk), np.array(ov)
    for head, row in enumerate(rows):
        energy = (ov[head, 1:].reshape(56, 2, 128)**2).sum(axis=(1, 2))
        k = row["pattern_frequency"]
        row["ov_energy_at_pattern_frequency"] = float(energy[k-1]/energy.sum())
        row["ov_energy_at_all_keys"] = float(energy[frequencies-1].sum()/energy.sum())
    pair_rows = []
    for first in range(4):
        for second in range(first+1, 4):
            a, b = ov[first, 1:].ravel(), ov[second, 1:].ravel()
            pair_rows.append({"head_a": first, "head_b": second,
                              "ov_cosine_similarity": float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)),
                              "pattern_sum_rms_error_from_one": float(np.sqrt(np.mean((attention[:, first, 0]+attention[:, second, 0]-1)**2)))})
    return {"rows": rows, "pairs": pair_rows, "qk": qk, "ov": ov, "frequencies": frequencies,
            "coefficients": np.array(polynomial), "linear_coefficients": np.array(linear_coefficients),
            "pattern_coefficients": np.array(patterns), "best": np.array(best), "heads": heads, "predicted": predicted,
            "fit_ids": fit, "validation_ids": validation}


def plot_results(result):
    """Plot head-fit comparisons and separate sine/cosine weight spectra for all four heads."""
    paper_style()
    positions = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(positions-0.2, [r["quadratic_r2_validation"] for r in result["rows"]], width=0.4, color=SIN_COLOR, label="One-frequency quadratic")
    ax.bar(positions+0.2, [r["all_key_linear_r2_validation"] for r in result["rows"]], width=0.4, color=COS_COLOR, label="All-key linear")
    ax.set_xticks(positions, [f"Head {j}" for j in positions])
    ax.set(ylabel="Fit-validation R²", title="Attention output approximations")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=2)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(HERE / "fit-quality.png", dpi=180)
    plt.close(fig)

    frequency = np.arange(1, 57)
    fig, axes = plt.subplots(4, 2, figsize=(12, 11))
    for head in range(4):
        for col, values, title in [(0, abs(result["qk"][head]), "QK token-score coefficients"),
                                   (1, np.linalg.norm(result["ov"][head], axis=1), "OV component norms")]:
            ax = axes[head, col]
            ax.bar(frequency-0.2, values[2::2], width=0.4, color=SIN_COLOR, label="sin")
            ax.bar(frequency+0.2, values[1::2], width=0.4, color=COS_COLOR, label="cos")
            ax.set(title=f"Head {head}: {title}", xlabel="Frequency", ylabel="Magnitude")
            if head == 0:
                ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "weight-spectra.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    ids = result["validation_ids"]
    for head, ax in enumerate(axes):
        actual, predicted = result["heads"][ids, head], result["predicted"][ids, head]
        ax.hexbin(actual.ravel(), predicted.ravel(), gridsize=45, mincnt=1, bins="log", cmap="Blues")
        low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        ax.plot([low, high], [low, high], "--", color="gray")
        ax.set(title=f"Head {head}, k={result['rows'][head]['quadratic_frequency']}", xlabel="Actual OV output", ylabel="Polynomial output")
    fig.tight_layout()
    fig.savefig(HERE / "head-fits.png", dpi=180)
    plt.close(fig)


def save_results(result):
    """Save head fits, pair comparisons, complete spectra, and fitting provenance."""
    for name, rows in [("fits.csv", result["rows"]), ("pairs.csv", result["pairs"])]:
        with (HERE / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(HERE / "fits.npz", **{key: result[key] for key in
                        ["qk", "ov", "frequencies", "coefficients", "linear_coefficients", "pattern_coefficients", "best", "fit_ids", "validation_ids"]})
    (HERE / "metrics.json").write_text(json.dumps({"heads": result["rows"], "pairs": result["pairs"]}, indent=2, allow_nan=False)+"\n")
    for row in result["rows"]:
        print("Head:", row)


def main():
    """Analyze the attention outputs before plotting and saving."""
    result = analyze()
    plot_results(result)
    save_results(result)


if __name__ == "__main__":
    main()

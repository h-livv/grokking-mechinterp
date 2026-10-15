"""Section 4.2 activation signals: calculate sum fits, then plot and save them.

Run: python 4.2-weights/activations/analyze.py
Only the discovery half of development data determines regression coefficients.
The other half assesses fit quality; full-grid plots are descriptive.
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
from shared import (load_model, readout_components, collect_activations, fit_partitions,
                    r_squared, paper_style, SIN_COLOR, COS_COLOR)


def analyze():
    """Test output-linked h directions against matched sum features and four products.

    q_cos = h @ u_k and q_sin = h @ v_k use Eq. 2's unnormalized cosine/sine
    convention. A matched fit is gain*cos(k(a+b)) or gain*sin(k(a+b)), with no
    intercept. The product fit uses [cos(a)cos(b), sin(a)sin(b), sin(a)cos(b),
    cos(a)sin(b)] at the selected frequency, as in the paper's Table 1.
    """
    model, inputs = load_model()
    weights = readout_components(model)
    hidden, _, _ = collect_activations(model, inputs)
    inputs = inputs.numpy()
    partitions = fit_partitions()
    fit, validation = partitions["discovery"], partitions["validation"]
    frequencies = weights["frequencies"]
    angles_a = inputs[:, 0, None] * (2*np.pi*frequencies/113)
    angles_b = inputs[:, 1, None] * (2*np.pi*frequencies/113)
    ca, sa, cb, sb = np.cos(angles_a), np.sin(angles_a), np.cos(angles_b), np.sin(angles_b)
    products = np.stack([ca*cb, sa*sb, sa*cb, ca*sb], axis=-1)
    sums = np.stack([np.cos(angles_a+angles_b), np.sin(angles_a+angles_b)], axis=-1)
    # These directions are fixed by the learned output weights, not fitted as a free probe.
    signals = (hidden @ weights["directions"].T).reshape(len(inputs), len(frequencies), 2)
    gains = np.empty((len(frequencies), 2))
    product_coefficients = np.empty((len(frequencies), 4, 2))
    predicted = np.empty_like(signals)
    rows = []
    for index, frequency in enumerate(frequencies):
        # Fit on discovery IDs only; all other rows are predictions from these fixed coefficients.
        coefficients, _, rank, _ = np.linalg.lstsq(products[fit, index], signals[fit, index], rcond=None)
        assert rank == 4, "The discovery product design must identify all four terms."
        product_coefficients[index] = coefficients
        product_prediction = products[:, index] @ coefficients
        for component, name in enumerate(("cos", "sin")):
            feature = sums[:, index, component]
            signal = signals[:, index, component]
            # Match cos to cos and sin to sin; allowing an arbitrary phase would test a weaker claim.
            gain = np.linalg.lstsq(feature[fit, None], signal[fit], rcond=None)[0][0]
            gains[index, component] = gain
            predicted[:, index, component] = gain * feature
            row = {"frequency": int(frequency), "component": name, "gain": float(gain),
                   "cc": float(coefficients[0, component]), "ss": float(coefficients[1, component]),
                   "sc": float(coefficients[2, component]), "cs": float(coefficients[3, component]),
                   "r2_discovery": r_squared(signal[fit], predicted[fit, index, component]),
                   "r2_validation": r_squared(signal[validation], predicted[validation, index, component]),
                   "r2_full_grid": r_squared(signal, predicted[:, index, component]),
                   "product_r2_validation": r_squared(signal[validation], product_prediction[validation, component])}
            rows.append(row)
    return {"inputs": inputs, "signals": signals, "predicted": predicted, "gains": gains,
            "products": product_coefficients, "frequencies": frequencies, "rows": rows,
            "fit_ids": fit, "validation_ids": validation}


def plot_results(result):
    """Plot each fixed frequency's observed/fitted signals and the Table 1 product coefficients."""
    paper_style()
    p = 113
    validation = result["validation_ids"]
    for index, frequency in enumerate(result["frequencies"]):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        observed, predicted = result["signals"][:, index], result["predicted"][:, index]
        bound = max(abs(observed).max(), abs(predicted).max())
        for component, name in enumerate(("cos", "sin")):
            for column, (values, title) in enumerate(((observed, "Observed"), (predicted, "Fitted sum feature"))):
                image = axes[component, column].imshow(values[:, component].reshape(p, p).T,
                            origin="upper", cmap="RdBu", vmin=-bound, vmax=bound, interpolation="nearest")
                axes[component, column].set(title=f"{title}: {name}, k={frequency}", xlabel="Operand a", ylabel="Operand b")
                fig.colorbar(image, ax=axes[component, column], label="Output-linked signal")
            ax = axes[component, 2]
            ax.scatter(observed[validation, component], predicted[validation, component], s=5, alpha=0.25,
                       color=COS_COLOR if component == 0 else SIN_COLOR)
            ax.plot([-bound, bound], [-bound, bound], color="gray", linestyle="--")
            score = result["rows"][2*index+component]["r2_validation"]
            ax.set(title=f"Fit-validation R² = {score:.3f}", xlabel="Observed signal", ylabel="Fitted signal")
        fig.tight_layout()
        fig.savefig(HERE / f"frequency-{frequency}.png", dpi=180)
        plt.close(fig)

    frequencies = result["frequencies"]
    positions = np.arange(len(frequencies))
    quality = np.array([row["r2_validation"] for row in result["rows"]]).reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(positions-0.2, quality[:, 1], width=0.4, color=SIN_COLOR, label="sin")
    ax.bar(positions+0.2, quality[:, 0], width=0.4, color=COS_COLOR, label="cos")
    ax.set_xticks(positions, frequencies)
    ax.set(xlabel="Frequency k", ylabel="Fit-validation R²", title="Matched sum-feature fits")
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=1, label="90% reference")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3)
    fig.tight_layout()
    fig.savefig(HERE / "fit-quality.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(frequencies), figsize=(13, 6), squeeze=False)
    for index, frequency in enumerate(frequencies):
        for component, name in enumerate(("cos", "sin")):
            ax = axes[component, index]
            ax.bar(["cc", "ss", "sc", "cs"], result["products"][index, :, component],
                   color=COS_COLOR if component == 0 else SIN_COLOR)
            ax.axhline(0, color="gray", linewidth=0.7)
            ax.set(title=f"{name} readout, k={frequency}", ylabel="Fitted coefficient")
    fig.tight_layout()
    fig.savefig(HERE / "product-coefficients.png", dpi=180)
    plt.close(fig)


def save_results(result):
    """Save the Table 1 analogue, frozen coefficients, and full-grid observed signals."""
    with (HERE / "fits.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result["rows"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["rows"])
    np.savez_compressed(HERE / "signals.npz", signals=result["signals"], gains=result["gains"],
                        product_coefficients=result["products"], frequencies=result["frequencies"],
                        fit_ids=result["fit_ids"], validation_ids=result["validation_ids"])
    for row in result["rows"]:
        print(f"k={row['frequency']} {row['component']}: gain={row['gain']:.4f}, validation R²={row['r2_validation']:.4f}")


def main():
    """Analyze the output-linked signals before plotting and saving their fits."""
    result = analyze()
    plot_results(result)
    save_results(result)


if __name__ == "__main__":
    main()

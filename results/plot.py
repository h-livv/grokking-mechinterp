"""Redraw the recorded training trajectory; this script has no mechanism analysis.

Run: python results/plot.py
load_results() reads existing observations; plot_results() draws them;
update_checksums() records the changed image/source hashes in the manifest.
"""

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "4.1-periodicity"))
from common import paper_style


def load_results():
    """Return CSV rows and run metadata; do not train or evaluate the model."""
    manifest = json.loads((HERE / "manifest.json").read_text())
    with (HERE / "metrics.csv").open() as file:
        rows = list(csv.DictReader(file))
    return rows, manifest


def plot_results(rows, manifest):
    """Save the original accuracy/loss trajectory with the same reference annotations."""
    paper_style()
    updates = [int(row["step"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label, color in (("train", "Training", "blue"), ("dev", "Validation", "red")):
        axes[0].plot(
            updates, [float(row[key + "_accuracy"]) for row in rows], label=label, color=color,
        )
        axes[1].semilogy(
            updates, [float(row[key + "_nll"]) for row in rows], label=label, color=color,
        )
    axes[0].axhline(0.95, color="gray", linestyle="--", linewidth=1, label="95% threshold")
    axes[0].set(ylabel="Accuracy", xlabel="Training updates", ylim=(-0.02, 1.02))
    axes[1].set(ylabel="Mean cross-entropy (nats)", xlabel="Training updates")
    for ax in axes:
        ax.axvline(manifest["selected_step"], color="black", linestyle=":", linewidth=1)
        ax.legend()
        ax.grid()
    fig.tight_layout()
    fig.savefig(HERE / "training-trajectory.png", dpi=180)
    plt.close(fig)


def update_checksums(manifest):
    """Refresh only the plot/script hashes; leave weights and recorded metrics unchanged."""
    for name in ("training-trajectory.png", "plot.py"):
        digest = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        manifest["files_sha256"][f"results/{name}"] = digest
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    """Read the existing results, redraw their figure, and refresh output checksums."""
    rows, manifest = load_results()
    plot_results(rows, manifest)
    update_checksums(manifest)


if __name__ == "__main__":
    main()

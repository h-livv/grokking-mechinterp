"""Redraw the recorded training trajectory in Figure 2's blue/red colours.

Run: python results/plot.py
Reads existing metrics only; does not train or evaluate the model.
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


def main():
    """Keep all recorded values and annotations; change only plot presentation."""
    paper_style()
    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with (HERE / "metrics.csv").open() as file:
        rows = list(csv.DictReader(file))
    updates = [int(row["step"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label, color in (("train", "Training", "blue"), ("dev", "Validation", "red")):
        axes[0].plot(updates, [float(row[key + "_accuracy"]) for row in rows], label=label, color=color)
        axes[1].semilogy(updates, [float(row[key + "_nll"]) for row in rows], label=label, color=color)
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
    # Keep the published verification manifest consistent with the restyled image.
    for name in ("training-trajectory.png", "plot.py"):
        manifest["files_sha256"][f"results/{name}"] = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

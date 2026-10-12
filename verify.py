"""Verify the published bias-free run using CPU execution; no training is run.

Checks artifact hashes, the exact split, initial weights, the selected state,
aggregate metrics, development predictions, and shortcut-stratified results.
The reserved final split is checked for membership but never evaluated.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import modular_addition_split
from model import Transformer


@torch.no_grad()
def main():
    """Reconstruct the task and independently check the published evidence."""
    root = Path(__file__).resolve().parent
    results = root / "results"
    manifest = json.loads((results / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    inputs, targets, expected_split = modular_addition_split(
        manifest["architecture"]["modulus"], manifest["data_seed"])
    saved = np.load(results / "split.npz", allow_pickle=False)
    split = {name: saved[name] for name in ("train", "dev", "final")}
    for name in split:
        assert np.array_equal(split[name], expected_split[name]), name
        assert len(split[name]) == manifest["split_sizes"][name]
    assert np.array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(len(inputs)))

    torch.manual_seed(manifest["model_seed"])
    model = Transformer().eval().requires_grad_(False)
    initial = torch.load(results / "initial.pt", map_location="cpu", weights_only=True)
    for name, value in model.state_dict().items():
        assert torch.equal(value, initial[name]), name
    assert sum(p.numel() for p in model.parameters()) == manifest["parameter_count"]
    model.load_state_dict(torch.load(results / "selected.pt", map_location="cpu", weights_only=True))
    x, y = torch.from_numpy(inputs), torch.from_numpy(targets)
    dev_ids = split["dev"]
    logits = model(x[dev_ids])[:, -1]
    labels = y[dev_ids]
    predictions = logits.argmax(-1)
    nll = F.cross_entropy(logits.double(), labels, reduction="none")
    wrong = logits.double().clone()
    wrong.scatter_(1, labels[:, None], -torch.inf)
    margins = logits.double().gather(1, labels[:, None])[:, 0] - wrong.max(-1).values

    training = np.zeros(len(inputs), dtype=bool)
    training[split["train"]] = True
    reverse = inputs[dev_ids, 1] * 113 + inputs[dev_ids, 0]
    reverse_seen = training[reverse]
    diagonal = inputs[dev_ids, 0] == inputs[dev_ids, 1]
    masks = {"all": np.ones(len(dev_ids), dtype=bool), "reverse_seen_in_train": reverse_seen,
             "neither_order_in_train": ~reverse_seen, "diagonal": diagonal,
             "off_diagonal_neither_order_in_train": ~reverse_seen & ~diagonal}
    for name, mask in masks.items():
        expected = manifest["results"]["development"][name]
        correct = predictions[mask] == labels[mask]
        assert int(correct.sum()) == expected["correct"] and int(mask.sum()) == expected["count"]
        assert math.isclose(nll[mask].mean().item(), expected["nll"], rel_tol=1e-5, abs_tol=1e-10)

    with (results / "development_predictions.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [int(r["pair_id"]) for r in rows] == dev_ids.tolist()
    for i, row in enumerate(rows):
        assert [int(row[k]) for k in ("a", "b", "target", "predicted")] == [
            int(x[dev_ids[i], 0]), int(x[dev_ids[i], 1]), int(labels[i]), int(predictions[i])]
        assert (row["reverse_seen_in_train"] == "True") == bool(reverse_seen[i])
        assert (row["diagonal"] == "True") == bool(diagonal[i])
        assert math.isclose(float(row["nll"]), nll[i].item(), rel_tol=1e-5, abs_tol=1e-10)
        assert math.isclose(float(row["correct_margin"]), margins[i].item(), rel_tol=1e-5, abs_tol=1e-4)

    train_logits = model(x[split["train"]])[:, -1]
    assert (train_logits.argmax(-1) == y[split["train"]]).all()
    train_nll = F.cross_entropy(train_logits.double(), y[split["train"]]).item()
    assert math.isclose(train_nll, manifest["results"]["train"]["nll"], rel_tol=1e-5, abs_tol=1e-10)
    with (results / "metrics.csv").open() as file:
        history = list(csv.DictReader(file))
    assert [int(r["step"]) for r in history] == list(range(0, manifest["updates"] + 1, 100))
    for row in history:
        for name in ("train_nll", "dev_nll", "weight_norm_squared", "elapsed_seconds"):
            assert math.isfinite(float(row[name])) and float(row[name]) >= 0
        assert 0 <= float(row["train_accuracy"]) <= 1 and 0 <= float(row["dev_accuracy"]) <= 1
    best = min(history, key=lambda r: float(r["dev_nll"]))
    assert int(best["step"]) == manifest["selected_step"]
    assert math.isclose(float(best["train_nll"]), train_nll, rel_tol=1e-5, abs_tol=1e-10)
    assert math.isclose(float(best["dev_nll"]), nll.mean().item(), rel_tol=1e-5, abs_tol=1e-10)
    for partition, key in (("train", "first_logged_perfect_training_step"),
                           ("dev", "first_logged_perfect_development_step")):
        assert next(int(r["step"]) for r in history if float(r[partition + "_accuracy"]) == 1) == manifest[key]
    print("Verified hashes, exact split, initial weights, checkpoint selection, and recorded predictions.")
    print(f"Development: {int((predictions == labels).sum())}/{len(labels)}; NLL {nll.mean().item():.12g}")
    print("Neither order in training: 3123/3123. Final-set model evaluation was not performed.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

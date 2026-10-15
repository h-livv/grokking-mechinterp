"""Shared Section 4.2 calculations; reuse Section 4.1 loading, bases, and colours.

No plotting is performed here. Hooks observe the fixed model, and all fitting
uses a deterministic discovery half of the development split.
"""

from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "4.1-periodicity"))
from common import load_model, fourier_basis, paper_style, SIN_COLOR, COS_COLOR


def readout_components(model):
    """Compose G=U W_out and select the smallest frequency set covering 90% energy.

    The eight-pair cap and 90% threshold are local protocol choices. Selection
    uses only readout weights, not activation-fit scores or final-set accuracy.
    """
    U = model.unembed.weight.double().numpy()
    G = U @ model.mlp_out.weight.double().numpy()
    p = len(G)
    basis = fourier_basis(p)
    coefficients = basis.T @ G
    energy = (coefficients[1:].reshape((p - 1) // 2, 2, -1)**2).sum(axis=(1, 2))
    order = np.argsort(-energy)
    count = int(np.searchsorted(np.cumsum(energy[order]) / energy.sum(), 0.90) + 1)
    assert count <= 8, "The local sparse-readout criterion failed; do not assume a sparse mechanism."
    frequencies = np.sort(order[:count] + 1)
    indices = np.array([i for k in frequencies for i in (2*k - 1, 2*k)])
    selected = basis[:, indices] @ coefficients[indices]
    # Eq. 2 uses unnormalized cos/sin vectors. Convert the orthonormal coefficients accordingly.
    directions = np.sqrt(2 / p) * coefficients[indices]
    return {"U": U, "G": G, "basis": basis, "coefficients": coefficients,
            "energy": energy, "frequencies": frequencies, "indices": indices,
            "selected": selected, "directions": directions}


def collect_activations(model, inputs):
    """Return h[N,512], x1[N,128], and actual logits[N,113] at the '=' position.

    x1 is the input to mlp_in, and h is the output of ReLU. Hooks only observe;
    they return no replacements. Forward execution stays float32, as trained.
    """
    cache = {}

    def capture_input(module, args):
        """Record the residual state immediately before the MLP."""
        cache["x1"] = args[0][:, -1].detach()

    def capture_hidden(module, args, output):
        """Record all post-ReLU hidden neurons at the answer position."""
        cache["h"] = output[:, -1].detach()

    handles = [model.mlp_in.register_forward_pre_hook(capture_input),
               model.relu.register_forward_hook(capture_hidden)]
    hidden, residual, logits = [], [], []
    try:
        with torch.inference_mode():
            for tokens in inputs.split(256):
                z = model(tokens)[:, -1]
                hidden.append(cache["h"].double().numpy())
                residual.append(cache["x1"].double().numpy())
                logits.append(z.double().numpy())
    finally:
        for handle in handles:
            handle.remove()
    return np.concatenate(hidden), np.concatenate(residual), np.concatenate(logits)


def fit_partitions():
    """Split development IDs into 2,234 discovery and 2,235 fit-validation pairs.

    Seed 0 fixes the partition before fitting. The final split is never used
    for coefficients or predictive metrics, though full-grid diagnostics see it.
    """
    with np.load(ROOT / "results/split.npz", allow_pickle=False) as saved:
        train, dev = saved["train"], saved["dev"]
    shuffled = np.random.default_rng(0).permutation(dev)
    middle = len(shuffled) // 2
    discovery, validation = shuffled[:middle], shuffled[middle:]
    assert len(np.intersect1d(discovery, validation)) == 0
    return {"train": train, "discovery": discovery, "validation": validation, "development": dev}


def r_squared(actual, predicted):
    """Return 1-SSE/SST; negative values are valid and constant targets are undefined."""
    denominator = np.sum((actual - actual.mean())**2)
    if denominator == 0:
        return float("nan")
    return float(1 - np.sum((actual - predicted)**2) / denominator)


def score_logits(logits, inputs, ids):
    """Calculate accuracy and float64 NLL only on the explicitly supplied IDs."""
    selected = logits[ids]
    labels = (inputs[ids, 0] + inputs[ids, 1]) % logits.shape[1]
    loss = F.cross_entropy(torch.from_numpy(selected), torch.from_numpy(labels)).item()
    return float(np.mean(selected.argmax(axis=1) == labels)), loss


def cosine_scores(inputs, frequencies):
    """Return cos(2*pi*k*(a+b-c)/p) for every input, answer class, and selected k."""
    p = int(inputs[0, 2])
    difference = inputs[:, 0, None] + inputs[:, 1, None] - np.arange(p)[None, :]
    return np.cos(difference[:, :, None] * (2 * np.pi * frequencies / p))

"""Small shared helpers for the Section 4.1 plots; reuse model.py and data.py."""

import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import modular_addition_split
from model import Transformer

SIN_COLOR = "#636EFA"
COS_COLOR = "#EF553B"


def paper_style():
    """Use the pale background and blue/red palette of the paper's figures."""
    plt.rcParams.update({"axes.facecolor": "#E5ECF6", "axes.edgecolor": "white",
                         "axes.labelcolor": "#2A3F5F", "text.color": "#2A3F5F",
                         "xtick.color": "#2A3F5F", "ytick.color": "#2A3F5F",
                         "grid.color": "white", "grid.alpha": 1.0})


def load_model():
    """Load the frozen public weights and all ordered inputs, without labels."""
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    path = ROOT / "results/selected.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files_sha256"]["results/selected.pt"]
    torch.set_num_threads(1)
    model = Transformer()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    inputs, _, _ = modular_addition_split(
        p=manifest["architecture"]["modulus"], seed=manifest["data_seed"]
    )
    return model, torch.from_numpy(inputs)


def collect_grid(model, inputs):
    """Run the original forward pass; capture attention, neuron 0, and logits.

    Arrays use [a, b, feature]. Hooks only observe module outputs. The final
    attention row is reconstructed from the actual queries and keys because
    model.py computes softmax directly rather than through a named module.
    '=' can attend to all three positions, so this row needs no causal mask.
    """
    cache = {}
    handles = []
    for name in ("query", "key", "relu"):
        def capture(module, args, output, name=name):
            cache[name] = output.detach()
        handles.append(getattr(model, name).register_forward_hook(capture))
    attention, neurons, logits = [], [], []
    try:
        with torch.inference_mode():
            for tokens in inputs.split(256):
                z = model(tokens)[:, -1, :]
                q = cache["query"][:, -1].reshape(-1, 4, 32)
                k = cache["key"].reshape(-1, 3, 4, 32)
                scores = torch.einsum("bhd,bthd->bht", q, k) / np.sqrt(32)
                weights = scores.softmax(dim=-1)
                torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[:, :, 0]))
                attention.append(weights[:, :, 0].numpy())  # '=' attending to a.
                neurons.append(cache["relu"][:, -1, 0].numpy())
                logits.append(z.numpy())
    finally:
        for handle in handles:
            handle.remove()
    p = model.unembed.out_features
    return (np.concatenate(attention).reshape(p, p, 4),
            np.concatenate(neurons).reshape(p, p),
            np.concatenate(logits).reshape(p, p, p))


def fourier_basis(p):
    """Real orthonormal columns: constant, cos(1), sin(1), ..., cos(56), sin(56)."""
    assert p >= 3 and p % 2 == 1
    a = np.arange(p)
    columns = [np.ones(p) / np.sqrt(p)]
    for k in range(1, (p + 1) // 2):
        angle = 2 * np.pi * k * a / p
        columns.extend([np.sqrt(2 / p) * np.cos(angle),
                        np.sqrt(2 / p) * np.sin(angle)])
    basis = np.column_stack(columns)
    np.testing.assert_allclose(basis.T @ basis, np.eye(p), atol=3e-14)
    return basis


def fourier2d(values):
    """Transform only operand axes; separate sine/cosine cells give 2x2 blocks."""
    p = values.shape[0]
    assert values.shape[1] == p
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all()
    basis = fourier_basis(p)
    coefficients = np.einsum("ai,ab...,bj->ij...", basis, values, basis, optimize=True)
    restored = np.einsum("ai,ij...,bj->ab...", basis, coefficients, basis, optimize=True)
    np.testing.assert_allclose(restored, values, atol=1e-10)
    np.testing.assert_allclose(np.sum(coefficients**2), np.sum(values**2), rtol=1e-12)
    return coefficients


def fourier_spectrum(values):
    """Norms per real component and energy per paired frequency of a row table."""
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all()
    basis = fourier_basis(len(values))
    coefficients = basis.T @ values
    np.testing.assert_allclose(basis @ coefficients, values, atol=1e-12)
    norms = np.linalg.norm(coefficients, axis=1)
    energy = np.r_[norms[0]**2, (norms[1:].reshape(-1, 2)**2).sum(axis=1)]
    np.testing.assert_allclose(energy.sum(), np.sum(values**2), rtol=1e-12)
    return norms, energy


def draw_map(ax, values, title, fourier=False):
    """Use Figure 4's blue maps and explicit sine/cosine Fourier labels."""
    options = {"cmap": "Blues"}
    if fourier:
        maximum = max(float(values.max()), np.finfo(float).eps)
        options = {"cmap": "RdBu", "norm": TwoSlopeNorm(
            vmin=-maximum / 10, vcenter=0, vmax=maximum)}
    image = ax.imshow(values.T, origin="upper", interpolation="nearest", **options)
    ax.set_title(title)
    if fourier:
        components = ["Const"] + [f"{part} {k}" for k in range(1, (len(values) + 1) // 2)
                                    for part in ("cos", "sin")]
        ticks = np.arange(0, len(values), 9)
        labels = [components[i] for i in ticks]
        ax.set_xticks(ticks, labels, rotation=90, fontsize=8)
        ax.set_yticks(ticks, labels, fontsize=8)
        ax.set(xlabel="Fourier component of a", ylabel="Fourier component of b")
    else:
        ax.set(xlabel="Operand a", ylabel="Operand b")
    return image

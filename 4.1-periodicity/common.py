"""Shared Section 4.1 operations, separated by purpose.

Loading/capture: load_model(), collect_grid().
Mathematics: fourier_basis(), fourier2d(), fourier_spectrum(), frequency_shares().
Plotting only: paper_style(), draw_map().
The Fourier functions return numbers; they neither draw nor save figures.
"""

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


def load_model():
    """Return the frozen model and all [a,b,=] inputs; no labels are used.

    The model runs in float32 on CPU. The returned input order has a as the
    outer index and b as the inner index, allowing reshape to [a,b,...].
    """
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    path = ROOT / "results/selected.pt"
    expected = manifest["files_sha256"]["results/selected.pt"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    torch.set_num_threads(1)
    model = Transformer()
    weights = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval().requires_grad_(False)
    inputs, _, _ = modular_addition_split(
        p=manifest["architecture"]["modulus"],
        seed=manifest["data_seed"],
    )
    return model, torch.from_numpy(inputs)


def collect_grid(model, inputs):
    """Return attention [a,b,head], neuron-0 activation [a,b], and logits [a,b,c].

    Hooks record query, key, and post-ReLU tensors. Attention from '=' to a is
    reconstructed from these queries/keys, including all three keys in softmax.
    The final position sees every token, so its attention row needs no mask.
    Full-grid inputs include the reserved final partition; no metrics are used.
    """
    cache = {}
    handles = []
    for name in ("query", "key", "relu"):
        def capture(module, args, output, name=name):
            """Record a tensor without returning a replacement for the forward pass."""
            # Observe only: returning a tensor here would replace the module output.
            cache[name] = output.detach()
        handles.append(getattr(model, name).register_forward_hook(capture))
    attention, neurons, logits = [], [], []
    try:
        with torch.inference_mode():
            for tokens in inputs.split(256):
                z = model(tokens)[:, -1, :]
                q = cache["query"][:, -1].reshape(-1, 4, 32)  # Final-position queries: [batch, head, 32].
                # All keys: [batch, token, head, 32], including the = token.
                k = cache["key"].reshape(-1, 3, 4, 32)
                # Contract head features, keeping batch/head/token separate; scale by
                # sqrt(d_head).
                scores = torch.einsum("bhd,bthd->bht", q, k) / np.sqrt(32)
                weights = scores.softmax(dim=-1)  # Normalize over all three keys, not just the operands.
                torch.testing.assert_close(
                    weights.sum(-1), torch.ones_like(weights[:, :, 0]),
                )
                attention.append(weights[:, :, 0].numpy())  # Key index 0 is a; the query position is =.
                # Read neuron 0 after ReLU, at the answer position.
                neurons.append(cache["relu"][:, -1, 0].numpy())
                logits.append(z.numpy())
    finally:
        for handle in handles:
            handle.remove()
    p = model.unembed.out_features
    return (
        # Input order is a-major, b-minor, so this becomes [a,b,head].
        np.concatenate(attention).reshape(p, p, 4),
        np.concatenate(neurons).reshape(p, p),
        np.concatenate(logits).reshape(p, p, p),
    )


def fourier_basis(p):
    """Return B[p,p] with columns constant, cos(1), sin(1), ..., cos(56), sin(56).

    Sine/cosine columns have normalization sqrt(2/p), making B.T @ B = I.
    """
    assert p >= 3 and p % 2 == 1  # This real basis pairs every nonzero frequency for odd p.
    a = np.arange(p)
    columns = [np.ones(p) / np.sqrt(p)]
    for k in range(1, (p + 1) // 2):
        angle = 2 * np.pi * k * a / p
        columns.extend([  # Keep cosine and sine adjacent; their combined energy is phase-independent.
            np.sqrt(2 / p) * np.cos(angle),
            np.sqrt(2 / p) * np.sin(angle),
        ])
    basis = np.column_stack(columns)
    np.testing.assert_allclose(basis.T @ basis, np.eye(p), atol=3e-14)
    return basis


def fourier2d(values):
    """Transform values[a,b,...] to coefficients[i,j,...], leaving features intact.

    For a scalar map this is B.T @ values @ B. Extra axes, such as heads or
    answer classes, remain unchanged. Reconstruction and energy must agree.
    """
    p = values.shape[0]
    assert values.shape[1] == p
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all()
    basis = fourier_basis(p)
    coefficients = np.einsum(  # Apply B.T to both operand axes; trailing feature/answer axes stay intact.
        "ai,ab...,bj->ij...", basis, values, basis, optimize=True,
    )
    restored = np.einsum(  # Inverse basis change: the original observations should be recovered.
        "ai,ij...,bj->ab...", basis, coefficients, basis, optimize=True,
    )
    np.testing.assert_allclose(restored, values, atol=1e-10)
    np.testing.assert_allclose(
        np.sum(coefficients**2), np.sum(values**2), rtol=1e-12,
    )
    return coefficients


def fourier_spectrum(values):
    """Return component norms and paired energies for values[residue,feature].

    norms has p entries in basis order. energy has one constant entry followed
    by one cos²+sin² energy per positive frequency; it sums to ||values||_F².
    """
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all()
    basis = fourier_basis(len(values))
    coefficients = basis.T @ values  # Transform residue rows, preserving the feature coordinates.
    np.testing.assert_allclose(basis @ coefficients, values, atol=1e-12)
    # Pool over features; retain a separate norm for each real basis component.
    norms = np.linalg.norm(coefficients, axis=1)
    energy = np.r_[  # Pair cos² + sin² at each frequency, keeping the constant entry separate.
        norms[0]**2, (norms[1:].reshape(-1, 2)**2).sum(axis=1),
    ]
    np.testing.assert_allclose(energy.sum(), np.sum(values**2), rtol=1e-12)
    return norms, energy


def frequency_shares(coefficients):
    """Return shares for cells where either operand has frequency k.

    Remove the input-independent mean, sum the relevant rows and columns, and
    subtract their overlap once. Different frequencies' shares can overlap.
    """
    energy = coefficients**2
    energy[0, 0] = 0  # Measure variation around the mean, rather than including the mean itself.
    shares = []
    for k in range(1, (len(energy) + 1) // 2):
        pair = slice(2 * k - 1, 2 * k + 1)  # Basis indices 2k-1 and 2k are cos(k) and sin(k).
        amount = (  # Include frequency k in either operand: rows + columns - their intersection.
            energy[pair, :].sum() + energy[:, pair].sum() - energy[pair, pair].sum()
        )
        shares.append(amount / energy.sum())
    return np.asarray(shares)


def paper_style():
    """Style figures without loading data or changing any analysis values."""
    plt.rcParams.update({
        "axes.facecolor": "#E5ECF6", "axes.edgecolor": "white",
        "axes.labelcolor": "#2A3F5F", "text.color": "#2A3F5F",
        "xtick.color": "#2A3F5F", "ytick.color": "#2A3F5F",
        "grid.color": "white", "grid.alpha": 1.0,
    })


def draw_map(ax, values, title, fourier=False):
    """Draw a horizontally and b vertically, with explicit Fourier component labels.

    Fourier maps use the paper's zero-centred palette. Magnitudes are
    nonnegative, so their data occupy its white-to-blue half.
    """
    options = {"cmap": "Blues"}
    if fourier:
        maximum = max(float(values.max()), np.finfo(float).eps)
        options = {
            "cmap": "RdBu",
            "norm": TwoSlopeNorm(vmin=-maximum / 10, vcenter=0, vmax=maximum),
        }
    image = ax.imshow(
        values.T, origin="upper", interpolation="nearest", **options,
    )
    ax.set_title(title)
    if fourier:
        components = ["Const"] + [
            f"{part} {k}" for k in range(1, (len(values) + 1) // 2) for part in ("cos", "sin")
        ]
        ticks = np.arange(0, len(values), 9)
        labels = [components[i] for i in ticks]
        ax.set_xticks(ticks, labels, rotation=90, fontsize=8)
        ax.set_yticks(ticks, labels, fontsize=8)
        ax.set(xlabel="Fourier component of a", ylabel="Fourier component of b")
    else:
        ax.set(xlabel="Operand a", ylabel="Operand b")
    return image

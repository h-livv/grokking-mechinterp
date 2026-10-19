"""Section 4.3 mathematics: single-frequency quadratic fits, without plotting.

Reuse Section 4.2's frozen model, frequencies, capture, and development split.
Feature selection and coefficients use discovery pairs only.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "4.2-weights"))
from shared import (load_model, readout_components, collect_activations, fit_partitions,
                    r_squared, paper_style, SIN_COLOR, COS_COLOR)

TERMS = ["1", "cos a", "sin a", "cos b", "sin b", "cos 2a", "sin 2a",
         "cos 2b", "sin 2b", "cos a cos b", "sin a sin b", "sin a cos b", "cos a sin b"]


def fixed_readout(model):
    """Recompute the weight-selected frequencies and check the saved Section 4.2 choices."""
    weights = readout_components(model)
    saved = json.loads((ROOT / "4.2-weights/frequencies.json").read_text())
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    assert saved["checkpoint_sha256"] == manifest["files_sha256"]["results/selected.pt"]
    np.testing.assert_array_equal(weights["frequencies"], saved["frequencies"])
    return weights


def quadratic_features(inputs, k):
    """Return 13 independent terms spanning degree <=2 polynomials in ca,sa,cb,sb.

    cos(2a), sin(2a), etc. replace redundant squares: ca²+sa²=cb²+sb²=1.
    All angles include 2*pi*k/p. Columns 9:13 are the four cross-input products.
    """
    p = int(inputs[0, 2])
    a, b = inputs[:, :2].T * (2*np.pi*k/p)
    ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
    return np.column_stack([np.ones(len(inputs)), ca, sa, cb, sb,
                            np.cos(2*a), np.sin(2*a), np.cos(2*b), np.sin(2*b),
                            ca*cb, sa*sb, sa*cb, ca*sb])


def column_r2(actual, predicted):
    """Return one R² per output coordinate; undefined constant-coordinate fits are NaN."""
    variance = np.sum((actual-actual.mean(axis=0))**2, axis=0)
    return np.divide(variance-np.sum((actual-predicted)**2, axis=0), variance,
                     out=np.full_like(variance, np.nan), where=variance > 0)


def fit_quadratics(inputs, targets, frequencies, partitions):
    """Fit all candidates, select a frequency on discovery, then assess frozen choices.

    targets is [input, coordinate]. Return coefficients [13,coordinate] and the
    frequency index of each coordinate, plus per-frequency discovery/validation R².
    """
    fit, validation = partitions["discovery"], partitions["validation"]
    coefficients, discovery_r2, validation_r2 = [], [], []
    for k in frequencies:
        features = quadratic_features(inputs, k)
        coefficient, _, rank, _ = np.linalg.lstsq(features[fit], targets[fit], rcond=None)
        assert rank == len(TERMS)
        coefficients.append(coefficient)
        discovery_r2.append(column_r2(targets[fit], features[fit] @ coefficient))
        validation_r2.append(column_r2(targets[validation], features[validation] @ coefficient))
    discovery_r2, validation_r2 = np.array(discovery_r2), np.array(validation_r2)
    # Do not use validation scores to choose a frequency or eligible-neuron mask.
    best = np.argmax(np.nan_to_num(discovery_r2, nan=-np.inf), axis=0)
    columns = np.arange(targets.shape[1])
    selected = np.stack(coefficients)[best, :, columns].T
    return {"coefficients": selected, "best": best, "discovery_r2": discovery_r2,
            "validation_r2": validation_r2}


def predict_quadratics(inputs, frequencies, best, coefficients):
    """Evaluate frozen per-coordinate polynomials, grouping coordinates by frequency."""
    predicted = np.empty((len(inputs), coefficients.shape[1]))
    for index, k in enumerate(frequencies):
        selected = np.flatnonzero(best == index)
        predicted[:, selected] = quadratic_features(inputs, k) @ coefficients[:, selected]
    return predicted


def collect_heads(model, inputs):
    """Capture each head's output after its own OV map, before heads are summed.

    Hook the input of attention_out, reshape joined [N,128] into [N,4,32],
    and apply the corresponding columns of W_O. These are residual-stream
    contributions [N,4,128], not attention probabilities or unprojected values.
    Also return attention [N,4,3] from '=' to a, b, and '='. All three keys are
    included in the softmax; learned positions and self-attention stay intact.
    """
    cache = {}
    def capture(module, args):
        """Observe the concatenated attention-head outputs at the answer position."""
        cache["joined"] = args[0][:, -1].detach()
    handle = model.attention_out.register_forward_pre_hook(capture)
    def capture_projection(module, args, output, name):
        """Observe queries/keys so attention probabilities can be reconstructed."""
        cache[name] = output.detach()
    q_handle = model.query.register_forward_hook(lambda m, a, o: capture_projection(m, a, o, "q"))
    k_handle = model.key.register_forward_hook(lambda m, a, o: capture_projection(m, a, o, "k"))
    values, attention = [], []
    try:
        with torch.inference_mode():
            for tokens in inputs.split(256):
                model(tokens)
                joined = cache["joined"].reshape(-1, 4, 32).double().numpy()
                W_O = model.attention_out.weight.double().numpy().reshape(128, 4, 32)
                contributions = np.einsum("nhd,ohd->nho", joined, W_O)
                np.testing.assert_allclose(contributions.sum(axis=1),
                                           joined.reshape(-1, 128) @ W_O.reshape(128, 128).T, atol=1e-11)
                values.append(contributions)
                q = cache["q"].reshape(-1, 3, 4, 32).transpose(1, 2)
                k = cache["k"].reshape(-1, 3, 4, 32).transpose(1, 2)
                scores = q @ k.transpose(-2, -1) / np.sqrt(32)
                probabilities = scores[:, :, -1].softmax(-1).double().numpy()
                np.testing.assert_allclose(probabilities.sum(axis=-1), 1, atol=2e-7)
                attention.append(probabilities)
    finally:
        for item in (handle, q_handle, k_handle):
            item.remove()
    return np.concatenate(values), np.concatenate(attention)

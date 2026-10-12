# Modular-addition grokking

## Aim

Reproduce and adapt the small-transformer modular-addition experiment in [*Progress Measures for Grokking via Mechanistic Interpretability*](https://arxiv.org/html/2301.05217v3). Study delayed generalization and investigate whether the trained model implements the paper's proposed Fourier-based computation.

## Current results

- Implemented a one-block, bias-free PyTorch Transformer for addition modulo 113.
- Completed one run of **40,000 full-batch training updates** (model seed `999`, data seed `0`).
- Observed **behavioral grokking**: training accuracy first reached 100% at logged update **1,100**, while development accuracy first reached 100% at logged update **25,100**.
- The selected checkpoint at update **39,800** achieves **100% development accuracy (4,469/4,469)** with mean negative log-likelihood (NLL) of **4.10 × 10⁻⁷ nats**.
- The selected checkpoint also achieves **100% accuracy on 3,123/3,123 development pairs for which neither `(a, b)` nor `(b, a)` appeared in training**. A reverse-pair lookup baseline achieves 30.54% overall development accuracy.

![Training and development trajectory](results/training-trajectory.png)

[Core evidence and verification](results/README.md) · [Detailed findings](docs/findings.md)

These results establish delayed generalization in one bias-free run. **The model's internal computation has not yet been verified.**

## Remaining work

- **Spectral analysis:** Identify Fourier structure in the token embeddings and effective MLP readout; determine whether a sparse set of frequencies dominates.
- **Mechanistic analysis:** Test whether output-linked activation directions encode sum-dependent sine and cosine features.
- **Causal interventions:** Remove and retain candidate hidden subspaces during the forward pass, with rank-matched controls, to test their contribution to predictions.
- **Final evaluation:** Freeze the analysis procedure and evaluate the reserved final split.
- **Robustness:** Repeat the experiment across additional model seeds to assess whether the findings persist across training runs.

The objective is to test whether this model implements the Fourier multiplication algorithm—not to assume that behavioral grokking implies that it does.
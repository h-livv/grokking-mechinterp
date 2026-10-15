# Section 4.2: Composing model weights

These analyses follow [Section 4.2 of *Progress measures for grokking via
mechanistic interpretability*](https://arxiv.org/html/2301.05217v3#S4.SS2).
They connect the trained readout weights to sum-dependent activation signals
and test a compact approximation of the model's answer scores.

| Analysis | Result |
|---|---|
| [Readout composition](readout/README.md) | Ten SVD directions retain 99.9993% of readout energy; the selected Fourier directions have 5.99% norm error |
| [Activation signals](activations/README.md) | Seven of ten matched sine/cosine sum fits exceed 90% fit-validation R²; frequency 31 is weaker |
| [Logit approximation](logits/README.md) | Five cosine terms explain 96.83% of fit-validation logit variance and preserve all 2,235 answers |

The evidence supports a substantial Fourier sum computation, but this checkpoint
does not match every strength-of-fit claim in the reference run. Causal use of
the identified components remains to be tested in Section 4.4.

## Run

From the repository root, with the existing requirements installed:

```bash
python 4.2-weights/readout/analyze.py
python 4.2-weights/activations/analyze.py
python 4.2-weights/logits/analyze.py
```

Each directory contains one script, numerical outputs, plots, and its analysis.
Every script separates `analyze()`, `plot_results()`, and `save_results()`.
[shared.py](shared.py) reuses the existing model, data, Fourier basis, and colours
from Section 4.1. Comments explain the mathematical choices and indexing.

## Fixed choices and scope

The checkpoint is update **39,800**, model seed 999, data seed 0. Frequencies
**19, 20, 21, 31, 49** are selected from $G=UW_{\mathrm{out}}$ by the smallest
set covering at least 90% of its nonconstant paired-frequency energy, capped at
eight pairs. This threshold is a local protocol choice. The rule and checkpoint
hash are saved in [frequencies.json](frequencies.json).

Regression coefficients use **2,234 development-discovery pairs**. A fixed seed-0
partition reserves the other **2,235 development pairs for fit validation**.
These pairs were excluded from coefficient fitting, but the development split
had already been used for checkpoint selection. This is not an independent
final-test estimate.

Full-grid signal plots and reconstruction diagnostics include all inputs,
including reserved final inputs already inspected in Section 4.1. No final-split
accuracy, loss, or intervention metrics are evaluated. Model weights stay fixed.

## Checks

Validation checks the readout's Fourier reconstruction and SVD error, the
normalization of output-linked directions, the exact residual/MLP identity,
regression coefficients against independent closed-form calculations, and saved
fit-validation metrics. The original run's evidence verifier also passes.

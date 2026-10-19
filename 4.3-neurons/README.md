# Section 4.3: Neurons and attention heads

These analyses follow [Section 4.3 of *Progress measures for grokking via
mechanistic interpretability*](https://arxiv.org/html/2301.05217v3#S4.SS3).
They test whether individual components compute simple functions at one
frequency and whether their output weights read the same frequency.

| Analysis | Finding |
|---|---|
| [MLP neurons](neurons/README.md) | 332/512 neurons pass the 85% discovery-fit threshold; 299 of those also pass on fit validation |
| [Readout localization](neurons/README.md#readout-localization) | All 332 selected neurons have their strongest output-frequency pair at their assigned input frequency |
| [Attention heads](attention/README.md) | Head outputs mix frequencies; single-frequency quadratic fit-validation R² ranges from 0.380 to 0.814 |

The checkpoint supports frequency-localized neuron clusters. The paper's
clean division of attention heads into single-frequency computation and
embedding amplification is less clear here. These are fits and weight analyses;
causal replacement tests in Section 4.4 remain future work.

## Run

From the repository root, with the existing requirements installed:

```bash
python 4.3-neurons/neurons/analyze.py
python 4.3-neurons/attention/analyze.py
```

Each directory contains a script, saved coefficients and measurements, plots,
and a short analysis. Scripts separate `analyze()`, `plot_results()`, and
`save_results()`, with comments on the mathematical steps.
[fitting.py](fitting.py) contains the shared quadratic basis and captures head
outputs; it reuses the existing model loading, activation capture, Fourier
basis, split, and plotting colours from Sections 4.1–4.2.

## Fixed choices and scope

Use the same checkpoint at update **39,800**, frequencies **19, 20, 21, 31, 49**,
and [saved selection provenance](../4.2-weights/frequencies.json).
Coefficients and frequency choices use **2,234 development-discovery pairs**.
The other **2,235 development pairs** assess the frozen fits. The **85%** neuron
threshold follows the paper, applied here on discovery data to define clusters.

Development data previously selected the checkpoint, so fit-validation scores
are not independent final-test estimates. Full-grid maps and head-pair
diagnostics include final inputs, as in Section 4.1. No final accuracy or loss
is evaluated. Weights are fixed and no intervention is performed.

## Checks

The quadratic design has rank 13 and is equivalent to the full 15-monomial
degree-two design after accounting for sine/cosine identities. Independent
checks reproduce neuron/head fits and frequency choices, compare localization
with a complex FFT, reconstruct head outputs directly from Q/K/V weights,
and confirm unchanged model predictions, weights, and hook cleanup.

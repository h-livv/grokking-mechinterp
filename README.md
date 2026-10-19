# Grokking and Mechanistic Interpretability

## Aim

Study grokking in a small modular-addition transformer and test the Fourier
computation proposed in [*Progress Measures for Grokking via Mechanistic
Interpretability*](https://arxiv.org/html/2301.05217v3).

## Current progress

The trained model exhibits grokking: it fits training examples early and
generalizes to unseen pairs much later. See the [grokking findings](docs/grokking.md).

The [Section 4.1 analyses](4.1-periodicity/README.md) are complete, showing Fourier
structure in the embeddings, readout, attention, neuron 0, and logits.

The [Section 4.2 analyses](4.2-weights/README.md) connect readout weights to
sum-dependent activation signals and a compact approximation of the logits.
The fits vary by frequency.

The [Section 4.3 analyses](4.3-neurons/README.md) examine individual neurons and
attention heads. Neuron clusters align with their output frequencies; head
computations mix frequencies more than the reference account suggests.
[Saved evidence and verification](results/README.md) are available. The checkpoint was selected by minimum validation loss; full-grid
diagnostics include final inputs, but final accuracy and loss remain unreported.

## Remaining work

Test the proposed computation through controlled interventions (Section 4.4),
then report final-split results. Repeat across training seeds to assess robustness.

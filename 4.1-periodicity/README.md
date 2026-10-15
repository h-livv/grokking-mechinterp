# Section 4.1: Periodicity

These analyses follow [Section 4.1 of *Progress measures for grokking via
mechanistic interpretability*](https://arxiv.org/html/2301.05217v3#S4.SS1).
They examine the selected model at update 39,800, using the existing public
weights and original forward computation.

| Analysis | What is plotted |
|---|---|
| [Embeddings](embeddings/README.md) | Separate sine/cosine norms before and after training; Figure 3 left |
| [Readout](readout/README.md) | Separate sine/cosine norms of the neuron-logit map; Figure 3 right |
| [Attention](attention/README.md) | Attention from `=` to the first operand, for all four heads |
| [MLP neuron](neurons/README.md) | Neuron 0's activation at `=` |
| [Logits](logits/README.md) | The two-dimensional operand Fourier spectrum of answer scores |

The same five frequencies dominate the embeddings, the neuron-logit map, and
the equal-frequency logit blocks. Attention heads and neuron 0 also show strong frequency structure.
These are descriptive observations; they do not establish the addition mechanism
or the causal role of a component.

## Run

From the repository root, with the existing requirements installed:

```bash
python 4.1-periodicity/embeddings/analyze.py
python 4.1-periodicity/readout/analyze.py
python 4.1-periodicity/attention/analyze.py
python 4.1-periodicity/neurons/analyze.py
python 4.1-periodicity/logits/analyze.py
```

Each script writes its outputs beside itself. The scripts share
[common.py](common.py) for loading, Fourier transforms, and plotting conventions. Model and data definitions
are reused from [model.py](../model.py) and [data.py](../data.py).

## Scope and checks

Attention, neuron, and logit maps use all **12,769 ordered pairs**, including
training, development, and reserved final inputs. This is the full-grid
diagnostic requested by the paper's workflow. No final accuracy, loss, or
intervention metrics are calculated, but the final inputs have now been
inspected; later analysis cannot describe them as an untouched holdout.

Heads 0–3 and neuron 0 were fixed before inspecting their maps. Frequencies
are measured from this checkpoint; none are copied from the paper's run.
No weights are changed and no interventions are performed.

The Fourier calculation checks orthonormality, inverse reconstruction, and
energy conservation. Additional validation compares hooked and original
logits, checks captured attention and neuron values, and compares frequency
energies with an independent complex FFT and known sum/difference signals.

## Section 4.1 completion

All three analyses in Section 4.1 are implemented: embedding spectra, attention
and neuron input maps, and the two-dimensional logit spectrum with its five
2x2 blocks. The readout spectrum completes the companion panel in Figure 3;
readout composition and sum-feature tests are now in [Section 4.2](../4.2-weights/README.md).
Causal intervention tests remain future work.

Plots use the paper's blue sine / red-orange cosine bars, white-to-blue
attention and activation maps, and a zero-centred red-white-blue Fourier scale.
Fourier norms are nonnegative, so they occupy the white-to-blue part of that
scale. Sine and cosine coordinates are kept separate in bars, labels, and
enlarged logit blocks.

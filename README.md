# Modular-addition grokking

## Aim

Reproduce and adapt the small-transformer modular-addition experiment in
[*Progress measures for grokking via mechanistic interpretability*](https://arxiv.org/html/2301.05217v3).
Study delayed generalization and investigate whether the model learns the
paper's Fourier-based computation.

## Current results

- Implemented a one-block, bias-free PyTorch transformer for addition modulo 113.
- Completed one run of **40,000 full-batch training updates**.
- Observed **behavioral grokking**: training accuracy reached 100% at update
  **1,100**, while development accuracy reached 100% at update **25,100**.
- The selected checkpoint at update **39,800** achieves **100% development
  accuracy (4,469/4,469)** with NLL **4.10e-7**, including **3,123/3,123** pairs
  for which neither operand order appeared in training. Reverse-pair lookup
  achieves 30.54% development accuracy.

![Training and development trajectory](results/training-trajectory.png)

[Core evidence and verification](results/README.md) · [Detailed findings](docs/findings.md).
These results establish delayed generalization in one bias-free run; the
internal mechanism has not yet been verified.

## Remaining work

- Analyze Fourier structure in embeddings and the effective MLP readout.
- Test candidate computations with controlled activation interventions.
- Evaluate the reserved final split after fixing the analysis choices.
- Repeat the experiment across additional seeds.

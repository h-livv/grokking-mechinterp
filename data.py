"""The modular-addition dataset and split from Step 2 of the experiment plan."""

import numpy as np


def modular_addition_split(p=113, seed=0):
    """Return NumPy inputs, targets, and train/dev/final indices.

    Each input is [a, b, p], with p denoting '='; the target is (a+b) % p.
    Use 30% of ordered pairs for training and halve the unused pairs.
    """
    a = np.repeat(np.arange(p, dtype=np.int64), p)
    b = np.tile(np.arange(p, dtype=np.int64), p)
    inputs = np.column_stack([a, b, np.full(p * p, p, dtype=np.int64)])
    targets = (a + b) % p
    permutation = np.random.default_rng(seed).permutation(p * p)
    n_train = int(0.30 * p * p)
    n_dev = (p * p - n_train) // 2
    split = {
        "train": permutation[:n_train],
        "dev": permutation[n_train:n_train + n_dev],
        "final": permutation[n_train + n_dev:],
    }
    assert len(np.unique(np.concatenate(list(split.values())))) == p * p
    assert np.all(targets == (inputs[:, 0] + inputs[:, 1]) % p)
    return inputs, targets, split

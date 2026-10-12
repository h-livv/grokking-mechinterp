# Core evidence: bias-free modular-addition run

One 40,000-update run, model seed 999 and data seed 0. The selected state is
update 39,800, chosen by minimum development NLL. Both MLP biases are disabled;
this evidence describes that adaptation of the reference experiment.

| File | Purpose |
|---|---|
| [metrics.csv](metrics.csv) | All 401 training/development evaluations |
| [training-trajectory.png](training-trajectory.png) | Accuracy and NLL curves from the CSV |
| [split.npz](split.npz) | Exact training/development/final indices, modulus, data seed |
| [selected.pt](selected.pt) | Selected model weights, without optimizer or private metadata |
| [initial.pt](initial.pt) | Initial weights for comparison with the trained state |
| [development_predictions.csv](development_predictions.csv) | Predictions, NLL, margins, and shortcut strata for all development cases |
| [manifest.json](manifest.json) | Run choices, software versions, scores, and artifact/source checksums |

## Verify

From the repository root, with the dependencies installed:

```bash
python verify.py
```

[verify.py](../verify.py) checks hashes, reconstructs the dataset using
[data.py](../data.py), loads the weights using [model.py](../model.py), and
checks the recorded predictions and aggregate results on CPU. The final
partition is audited for membership but is not evaluated by the model.
The corresponding training procedure is [train.py](../train.py).

The weight files are plain CPU float32 state dictionaries:

```python
import torch
from model import Transformer

model = Transformer()
model.load_state_dict(torch.load("results/selected.pt", map_location="cpu", weights_only=True))
model.eval().requires_grad_(False)
```

The full optimizer/checkpoint archive is unnecessary to verify the current
behavioral claims and is excluded from this compact evidence set. Individual
NLL checks use relative tolerance 1e-5 and absolute tolerance 1e-10 to allow
small float32-forward differences across execution environments.

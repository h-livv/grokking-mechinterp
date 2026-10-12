"""Full-batch modular-addition training. Edit the settings below, then run.

Each step makes one AdamW update using every training pair. Only the '='
position predicts the answer. Final-set pairs are saved but never evaluated.
"""

import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from data import modular_addition_split
from model import Transformer


DEVICE = "cuda"
STEPS = 40_000
SEED = 999
DATA_SEED = 0
OUT = Path("runs/baseline")


@torch.no_grad()
def evaluate(model, inputs, targets):
    """Answer-position mean cross-entropy and accuracy."""
    logits = model(inputs)[:, -1]
    loss = F.cross_entropy(logits.double(), targets).item()
    accuracy = (logits.argmax(-1) == targets).double().mean().item()
    return loss, accuracy


def main():
    """Build the data, train a fresh model, and save metrics and checkpoints."""
    torch.manual_seed(SEED)
    model = Transformer().to(DEVICE)
    OUT.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0,
                                 betas=(0.9, 0.98), eps=1e-8)

    # Generate the documented NumPy split, then convert it for PyTorch.
    inputs, targets, split = modular_addition_split(seed=DATA_SEED)
    inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
    split = {name: torch.from_numpy(ids) for name, ids in split.items()}
    torch.save({"inputs": inputs, "targets": targets, **split, "seed": SEED,
                "data_seed": DATA_SEED}, OUT / "data.pt")
    train_x, train_y = inputs[split["train"]].to(DEVICE), targets[split["train"]].to(DEVICE)
    dev_x, dev_y = inputs[split["dev"]].to(DEVICE), targets[split["dev"]].to(DEVICE)

    # Record the exact choices and source contents used for this run.
    source = Path(__file__).resolve().parent
    metadata = {
        "seed": SEED, "data_seed": DATA_SEED, "updates": STEPS,
        "architecture": {"modulus": 113, "blocks": 1, "width": 128,
                         "heads": 4, "head_width": 32, "mlp_width": 512,
                         "activation": "relu", "biases": False, "normalization": None,
                         "positions": "learned", "tied_embeddings": False},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "batch_size": len(train_y), "forward_dtype": "float32", "loss_dtype": "float64",
        "lr_schedule": "constant", "gradient_clipping": None,
        "optimizer_defaults": optimizer.defaults,
        "parameter_groups": [{**{k: v for k, v in group.items() if k != "params"},
                              "parameter_names": [name for name, _ in model.named_parameters()]}
                             for group in optimizer.param_groups],
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "device": DEVICE,
        "device_name": torch.cuda.get_device_name() if DEVICE.startswith("cuda") else "cpu",
        "source_revision": "SHA256 source snapshot (workspace has no Git repository)",
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in ("model.py", "data.py", "train.py")},
    }
    (OUT / "source").mkdir()
    for name in metadata["source_sha256"]:
        (OUT / "source" / name).write_bytes((source / name).read_bytes())
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if DEVICE.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    best_nll, best_step = float("inf"), 0
    crossed = set()

    def save(filename, step):
        # Save atomically so an interrupted write cannot masquerade as a checkpoint.
        numpy_state = np.random.get_state()
        state = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "torch_rng_state": torch.get_rng_state(),
                 "cuda_rng_states": torch.cuda.get_rng_state_all() if DEVICE.startswith("cuda") else [],
                 "numpy_rng_state": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                 "split": split, "metadata": metadata,
                 "elapsed_seconds": time.perf_counter() - start}
        temporary = OUT / (filename + ".tmp")
        torch.save(state, temporary)
        temporary.replace(OUT / filename)

    with (OUT / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "train_nll", "train_accuracy", "dev_nll", "dev_accuracy",
                         "weight_norm_squared", "elapsed_seconds", "checkpoint_identity"])
        for step in range(STEPS + 1):
            if step > 0:
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_x)[:, -1]
                # Float64 loss resolves small NLLs; the model stays float32.
                loss = F.cross_entropy(logits.double(), train_y)
                loss.backward()
                if step % 100 == 0:
                    assert torch.isfinite(loss).item(), f"Nonfinite loss at step {step}"
                    assert all(p.grad is not None and torch.isfinite(p.grad).all().item()
                               for p in model.parameters()), f"Nonfinite gradient at step {step}"
                optimizer.step()

            if step % 100 == 0 or step == STEPS:
                train_loss, train_accuracy = evaluate(model, train_x, train_y)
                dev_loss, dev_accuracy = evaluate(model, dev_x, dev_y)
                assert np.isfinite(train_loss) and np.isfinite(dev_loss)
                weight_norm = sum(p.detach().double().square().sum().item() for p in model.parameters())
                writer.writerow([step, train_loss, train_accuracy, dev_loss, dev_accuracy,
                                 weight_norm, time.perf_counter() - start, f"seed{SEED}-data{DATA_SEED}-step{step}"])
                file.flush()
                print(f"{step:5d}  train: {train_loss:.6g}, {train_accuracy:.1%}"
                      f"  dev: {dev_loss:.6g}, {dev_accuracy:.1%}", flush=True)
                if dev_loss < best_nll:
                    best_nll, best_step = dev_loss, step
                    save("best.pt", step)
                new_gates = {gate for gate in (0.25, 0.50, 0.75, 0.95)
                             if dev_accuracy >= gate} - crossed
                if new_gates:
                    save(f"transition_{step:05d}.pt", step)
                    crossed.update(new_gates)

            if step % 500 == 0 or step == STEPS:
                save(f"step_{step:05d}.pt", step)
    save("last.pt", STEPS)
    summary = {"updates": STEPS, "best_step": best_step, "best_dev_nll": best_nll,
               "elapsed_seconds": time.perf_counter() - start,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated() if DEVICE.startswith("cuda") else None,
               "peak_reserved_bytes": torch.cuda.max_memory_reserved() if DEVICE.startswith("cuda") else None}
    (OUT / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

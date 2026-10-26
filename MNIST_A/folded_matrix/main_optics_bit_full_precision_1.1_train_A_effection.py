"""Full-precision direct effective-matrix baseline for MNIST.

The hidden stages of the original 784->64->128->64->10 model are removed.
Instead, this script trains one continuous complex matrix

    K_eff[j, i] = a_eff[j, i] * exp(i * theta_eff[j, i])

with shape 10 x 784.  Its inference graph is intentionally the same as the
folded-single FP graph:

    phase encoding -> one complex matrix -> output normalization
    -> square-law detection -> optional log intensity.

The distinction is parameter provenance: this file learns K_eff directly,
whereas the folded-single PTQ file obtains K_eff by multiplying a trained
four-stage model.
"""

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DATA_DIR = "./data"
DOWNLOAD = True
EPOCHS = 100
BATCH_SIZE = 512
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA_LOSS = 1e-4
EARLY_STOPPING_MIN_DELTA_ACC = 1e-4
LR = 1e-2
LR_STEP_SIZE = 10
LR_GAMMA = 0.99
WEIGHT_DECAY = 0.0
VAL_RATIO = 0.1
NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2
USE_LOG_INTENSITY = True
EPS = 1e-12

N0 = 28 * 28
N4 = 10
SEEDS = [42, 123, 2024]
A_TRANS_INIT_VALUE = 1.0
A_TRANS_INIT_STD = 1e-3

CSV_SAVE_PATH = Path(
    "main_optics_continuous_effective_matrix_three_seed_results.csv"
)


class EffectiveSingleComplexMatrix(nn.Module):
    """A directly trainable full-precision K_eff in C^(10 x 784)."""

    def __init__(self, n0=N0, n4=N4, use_log_intensity=USE_LOG_INTENSITY):
        super().__init__()
        self.n0 = int(n0)
        self.n4 = int(n4)
        self.use_log_intensity = bool(use_log_intensity)

        self.path_theta_eff = nn.Parameter(torch.zeros(self.n4, self.n0))
        self.path_a_trans_eff = nn.Parameter(
            torch.clamp(
                A_TRANS_INIT_VALUE
                + A_TRANS_INIT_STD * torch.randn(self.n4, self.n0),
                0.0,
                1.0,
            )
        )

    def effective_matrix(self):
        """Return K_eff with shape [output=10, input=784]."""
        a_eff = torch.clamp(self.path_a_trans_eff, 0.0, 1.0)
        phase_eff = torch.complex(
            torch.cos(self.path_theta_eff),
            torch.sin(self.path_theta_eff),
        )
        return (a_eff * phase_eff).to(torch.complex64)

    @staticmethod
    def encode_input_phase(x):
        phase = math.pi * x.view(x.size(0), -1).to(torch.float32)
        return torch.complex(torch.cos(phase), torch.sin(phase)).to(torch.complex64)

    @staticmethod
    def normalize_field(psi):
        total_power = torch.sum(torch.abs(psi) ** 2, dim=1, keepdim=True)
        return psi / torch.sqrt(total_power + EPS)

    def complex_output(self, x):
        psi0 = self.encode_input_phase(x)
        psi4 = psi0 @ self.effective_matrix().T
        return self.normalize_field(psi4)

    def forward(self, x):
        intensity = torch.abs(self.complex_output(x)) ** 2
        if self.use_log_intensity:
            return torch.log(intensity + EPS)
        return intensity


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def configure_cuda():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def make_loader(dataset, batch_size, shuffle, device):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY and device.type == "cuda",
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(dataset, **kwargs)


def make_dataloaders(seed, device):
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        transform=transform,
        download=DOWNLOAD,
    )
    test_set = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        transform=transform,
        download=DOWNLOAD,
    )
    val_size = int(len(full_train) * VAL_RATIO)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=generator,
    )
    return (
        make_loader(train_set, BATCH_SIZE, True, device),
        make_loader(val_set, BATCH_SIZE, False, device),
        make_loader(test_set, BATCH_SIZE, False, device),
        torch.as_tensor(train_set.indices, dtype=torch.long),
        torch.as_tensor(val_set.indices, dtype=torch.long),
        torch.arange(len(test_set), dtype=torch.long),
    )


def target_probabilities(labels):
    return F.one_hot(labels, num_classes=N4).to(torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(
            F.log_softmax(logits, dim=1),
            target_probabilities(labels).to(device),
        )

        if training:
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.path_a_trans_eff.clamp_(0.0, 1.0)

        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch_size

    return total_loss / total, total_correct / total, total_correct, total


def save_checkpoint(
    model,
    seed,
    best_val_acc,
    best_val_epoch,
    final_epoch,
    stopped_early,
    test_metrics,
    train_indices,
    val_indices,
    test_indices,
):
    path = Path(
        f"main_optics_continuous_effective_matrix_model_seed_{seed}.pt"
    )
    test_loss, test_acc, correct, total = test_metrics
    with torch.no_grad():
        k_eff = model.effective_matrix().detach().cpu()

    torch.save(
        {
            "model_type": "direct_full_precision_effective_complex_matrix",
            "matrix_shape": (N4, N0),
            "n0": N0,
            "n4": N4,
            "use_log_intensity": USE_LOG_INTENSITY,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "k_eff": k_eff,
            "path_theta_eff": model.path_theta_eff.detach().cpu(),
            "path_a_trans_eff": model.path_a_trans_eff.detach().cpu(),
            "best_val_acc": best_val_acc,
            "best_val_epoch": best_val_epoch,
            "final_epoch": final_epoch,
            "stopped_early": stopped_early,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "correct": correct,
            "total": total,
            "seed": seed,
            "data_dir": str(Path(DATA_DIR).resolve()),
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "input_encoding": "psi0 = exp(i*pi*x)",
            "matrix_action": "psi4_raw = psi0 @ K_eff.T",
            "readout": (
                "normalize output field; square-law intensity; log intensity"
            ),
            "hidden_layers": 0,
            "hidden_bias_phases": 0,
            "parameterization": (
                "K_eff[j,i] = a_eff[j,i] * exp(i*theta_eff[j,i])"
            ),
            "folded_single_graph_compatible": True,
        },
        path,
    )
    return path


def train_one_seed(seed, device):
    set_seed(seed)
    (
        train_loader,
        val_loader,
        test_loader,
        train_indices,
        val_indices,
        test_indices,
    ) = make_dataloaders(seed, device)

    model = EffectiveSingleComplexMatrix().to(device)
    expected_params = 2 * N4 * N0
    actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual_params != expected_params:
        raise RuntimeError(
            f"Unexpected trainable parameter count: {actual_params} "
            f"(expected {expected_params})"
        )

    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA,
    )

    print("\n" + "=" * 90)
    print(f"Direct effective-matrix full precision | seed={seed}")
    print(f"Network: {N0} -> {N4}; K_eff shape: {N4} x {N0}")
    print(f"Trainable real scalars: {actual_params}")
    print("Epoch | Train loss | Train acc (%) | Val loss | Val acc (%)")

    best_val_acc = -1.0
    best_val_epoch = 0
    best_state = None
    best_stop_loss = float("inf")
    best_stop_acc = 0.0
    stale_epochs = 0
    stopped_early = False
    final_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        final_epoch = epoch
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        with torch.no_grad():
            val_loss, val_acc, _, _ = run_epoch(
                model, val_loader, criterion, device
            )

        print(
            f"{epoch:5d} | {train_loss:10.5f} | {100*train_acc:13.3f} | "
            f"{val_loss:8.5f} | {100*val_acc:11.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        loss_improved = (
            val_loss < best_stop_loss - EARLY_STOPPING_MIN_DELTA_LOSS
        )
        acc_improved = (
            val_acc > best_stop_acc + EARLY_STOPPING_MIN_DELTA_ACC
        )
        if loss_improved:
            best_stop_loss = val_loss
        if acc_improved:
            best_stop_acc = val_acc
        if loss_improved or acc_improved:
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()

        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            break

    if best_state is None:
        raise RuntimeError("No best validation state was recorded.")
    model.load_state_dict(best_state)
    with torch.no_grad():
        model.path_a_trans_eff.clamp_(0.0, 1.0)
        test_metrics = run_epoch(model, test_loader, criterion, device)

    checkpoint_path = save_checkpoint(
        model,
        seed,
        best_val_acc,
        best_val_epoch,
        final_epoch,
        stopped_early,
        test_metrics,
        train_indices,
        val_indices,
        test_indices,
    )
    test_loss, test_acc, correct, total = test_metrics
    print(
        f"Test accuracy: {100*test_acc:.6f}% ({correct}/{total}); "
        f"checkpoint: {checkpoint_path.resolve()}"
    )

    del train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "record_type": "seed_result",
        "seed": seed,
        "model": "direct_effective_matrix_fp",
        "matrix_shape": f"{N4}x{N0}",
        "trainable_real_scalars": actual_params,
        "best_val_epoch": best_val_epoch,
        "final_epoch": final_epoch,
        "stopped_early": stopped_early,
        "test_loss": test_loss,
        "test_acc_percent": 100.0 * test_acc,
        "correct": correct,
        "total": total,
        "checkpoint_path": str(checkpoint_path),
    }


def save_csv(rows):
    fieldnames = [
        "record_type",
        "seed",
        "model",
        "matrix_shape",
        "trainable_real_scalars",
        "best_val_epoch",
        "final_epoch",
        "stopped_early",
        "test_loss",
        "test_acc_percent",
        "mean_test_acc_percent",
        "sample_std_test_acc_percent",
        "correct",
        "total",
        "checkpoint_path",
    ]
    with CSV_SAVE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_all():
    configure_cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    rows = [train_one_seed(seed, device) for seed in SEEDS]

    accuracies = np.asarray(
        [row["test_acc_percent"] for row in rows],
        dtype=np.float64,
    )
    rows.append(
        {
            "record_type": "aggregate",
            "seed": "",
            "model": "direct_effective_matrix_fp",
            "matrix_shape": f"{N4}x{N0}",
            "trainable_real_scalars": 2 * N4 * N0,
            "best_val_epoch": "",
            "final_epoch": "",
            "stopped_early": "",
            "test_loss": "",
            "test_acc_percent": "",
            "mean_test_acc_percent": float(accuracies.mean()),
            "sample_std_test_acc_percent": float(accuracies.std(ddof=1)),
            "correct": "",
            "total": "",
            "checkpoint_path": "",
        }
    )
    save_csv(rows)
    print(f"CSV saved to: {CSV_SAVE_PATH.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="maximum epochs per seed"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="mini-batch size"
    )
    return parser.parse_args()


def main():
    global EPOCHS, BATCH_SIZE
    args = parse_args()
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    train_all()


if __name__ == "__main__":
    main()

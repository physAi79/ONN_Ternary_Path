"""Direct-single complex-matrix QAT baseline for MNIST.

This script removes all hidden stages and trains one 10 x 784 complex matrix
directly.  Its magnitude and phase are quantized during the forward pass with
an STE, so it is the direct-single QAT baseline described in the manuscript.
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


SHARED_RESULTS_CSV = Path(__file__).resolve().parent / "table6_single_matrix_comparison_results.csv"
SHARED_RESULTS_FIELDS = [
    "source_script", "dataset", "model", "construction", "record_type",
    "seed", "b_theta", "b_a", "test_accuracy_percent",
    "mean_accuracy_percent", "sample_std_accuracy_percent",
    "best_validation_epoch", "final_epoch", "parameter_count_real_scalars",
    "fold_scale", "fold_max_relative_error",
    "fold_relative_frobenius_error", "fold_prediction_mismatches",
    "correct", "total", "checkpoint_path",
]


def write_shared_results(rows):
    source_script = Path(__file__).name
    existing = []
    if SHARED_RESULTS_CSV.exists():
        with SHARED_RESULTS_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
            existing = [row for row in csv.DictReader(handle) if row.get("source_script") != source_script]
    normalized = []
    for row in rows:
        record = {field: "" for field in SHARED_RESULTS_FIELDS}
        record.update(row)
        record["source_script"] = source_script
        normalized.append(record)
    with SHARED_RESULTS_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHARED_RESULTS_FIELDS)
        writer.writeheader()
        writer.writerows(existing + normalized)


SOURCE_DIR = Path(__file__).resolve().parents[1] / "New_version_1.0" / "MNIST_A"
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = str(SOURCE_DIR / "data")
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
EPS = 1e-12

N0 = 28 * 28
N4 = 10
SEEDS = [42, 123, 2024]
B_THETA_VALUES = [1, 2]
B_ATTEN_VALUES = [1, 2]
A_TRANS_INIT_VALUE = 1.0
A_TRANS_INIT_STD = 1e-3

CSV_SAVE_PATH = Path(__file__).resolve().parent / (
    "direct_single_qat_three_seed_results_New_1012.csv"
)


class DirectSingleComplexMatrix(nn.Module):
    """One trainable complex map K in C^(10 x 784), with factorwise QAT."""

    def __init__(self, b_theta, b_atten, use_log_intensity=True):
        super().__init__()
        self.n0 = N0
        self.n4 = N4
        self.b_theta = int(b_theta)
        self.b_atten = int(b_atten)
        self.theta_levels = 2 ** self.b_theta
        self.atten_levels = 2 ** self.b_atten
        self.delta_theta = 2.0 * math.pi / self.theta_levels
        self.delta_atten = 1.0 / (self.atten_levels - 1)
        self.use_log_intensity = use_log_intensity

        self.path_theta = nn.Parameter(torch.zeros(N4, N0))
        self.path_a_trans = nn.Parameter(
            torch.clamp(
                A_TRANS_INIT_VALUE
                + A_TRANS_INIT_STD * torch.randn(N4, N0),
                0.0,
                1.0,
            )
        )

    def quantize_phase_ste(self, theta):
        theta_mod = torch.remainder(theta, 2.0 * math.pi)
        index = torch.round(theta_mod / self.delta_theta)
        index = torch.remainder(index, self.theta_levels)
        theta_q = self.delta_theta * index
        return theta + (theta_q - theta).detach()

    def quantize_atten_ste(self, a_trans):
        a_clamped = torch.clamp(a_trans, 0.0, 1.0)
        a_q = self.delta_atten * torch.round(a_clamped / self.delta_atten)
        a_q = torch.clamp(a_q, 0.0, 1.0)
        return a_trans + (a_q - a_trans).detach()

    def effective_matrix(self):
        theta_q = self.quantize_phase_ste(self.path_theta)
        a_q = self.quantize_atten_ste(self.path_a_trans)
        phase = torch.complex(torch.cos(theta_q), torch.sin(theta_q))
        return (a_q * phase).to(torch.complex64)

    @staticmethod
    def encode_input_phase(x):
        phase = math.pi * x.view(x.size(0), -1).to(torch.float32)
        return torch.complex(torch.cos(phase), torch.sin(phase)).to(torch.complex64)

    @staticmethod
    def normalize_field(psi):
        power = torch.sum(torch.abs(psi) ** 2, dim=1, keepdim=True)
        return psi / torch.sqrt(power + EPS)

    def forward(self, x):
        psi0 = self.encode_input_phase(x)
        psi4 = psi0 @ self.effective_matrix().T
        psi4 = self.normalize_field(psi4)
        intensity = torch.abs(psi4) ** 2
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
        root=DATA_DIR, train=True, transform=transform, download=DOWNLOAD
    )
    test_set = datasets.MNIST(
        root=DATA_DIR, train=False, transform=transform, download=DOWNLOAD
    )
    val_size = int(round(len(full_train) * VAL_RATIO))
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train, [train_size, val_size], generator=generator
    )
    return (
        make_loader(train_set, BATCH_SIZE, True, device),
        make_loader(val_set, BATCH_SIZE, False, device),
        make_loader(test_set, BATCH_SIZE, False, device),
        train_set.indices,
        val_set.indices,
    )


def make_target_probs(labels):
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
        loss = criterion(F.log_softmax(logits, dim=1), make_target_probs(labels).to(device))

        if training:
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.path_a_trans.clamp_(0.0, 1.0)

        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch_size

    return total_loss / total, total_correct / total, total_correct, total


def save_checkpoint(
    model,
    seed,
    b_theta,
    b_atten,
    best_val_acc,
    best_val_epoch,
    final_epoch,
    stopped_early,
    test_metrics,
    train_indices,
    val_indices,
):
    path = OUTPUT_DIR / (
        "direct_single_qat_"
        f"bt_{b_theta}_ba_{b_atten}_seed_{seed}.pt"
    )
    test_loss, test_acc, correct, total = test_metrics
    torch.save(
        {
            "model_type": "direct_single_complex_matrix_qat",
            "matrix_shape": (N4, N0),
            "n0": N0,
            "n4": N4,
            "b_theta": b_theta,
            "b_atten": b_atten,
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "best_val_acc": best_val_acc,
            "best_val_epoch": best_val_epoch,
            "final_epoch": final_epoch,
            "stopped_early": stopped_early,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "correct": correct,
            "total": total,
            "seed": seed,
            "data_dir": DATA_DIR,
            "train_indices": list(train_indices),
            "val_indices": list(val_indices),
            "input_encoding": "psi0 = exp(i*pi*x)",
            "readout": "normalize output field, then square-law intensity",
            "parameterization": "K_eff[j,i] = a[j,i] * exp(i*theta[j,i])",
        },
        path,
    )
    return path


def train_one(seed, b_theta, b_atten, loaders, indices, device):
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders
    train_indices, val_indices = indices
    model = DirectSingleComplexMatrix(b_theta, b_atten).to(device)
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
    )

    expected_params = 2 * N4 * N0
    actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual_params != expected_params:
        raise RuntimeError(
            f"Unexpected parameter count: {actual_params} != {expected_params}"
        )

    print("\n" + "=" * 90)
    print(
        f"Direct-single QAT | seed={seed} | b_theta={b_theta} | b_atten={b_atten}"
    )
    print(f"Matrix shape: {N4} x {N0}; trainable real scalars: {actual_params}")
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

        loss_improved = val_loss < best_stop_loss - EARLY_STOPPING_MIN_DELTA_LOSS
        acc_improved = val_acc > best_stop_acc + EARLY_STOPPING_MIN_DELTA_ACC
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
        raise RuntimeError("No best state was recorded.")
    model.load_state_dict(best_state)
    with torch.no_grad():
        model.path_a_trans.clamp_(0.0, 1.0)
        test_metrics = run_epoch(model, test_loader, criterion, device)

    path = save_checkpoint(
        model,
        seed,
        b_theta,
        b_atten,
        best_val_acc,
        best_val_epoch,
        final_epoch,
        stopped_early,
        test_metrics,
        train_indices,
        val_indices,
    )
    test_loss, test_acc, correct, total = test_metrics
    print(
        f"Test accuracy: {100*test_acc:.6f}% ({correct}/{total}); "
        f"checkpoint: {path.resolve()}"
    )
    return {
        "record_type": "seed_result",
        "seed": seed,
        "b_theta": b_theta,
        "b_atten": b_atten,
        "best_val_epoch": best_val_epoch,
        "final_epoch": final_epoch,
        "stopped_early": stopped_early,
        "test_loss": test_loss,
        "test_acc_percent": 100.0 * test_acc,
        "correct": correct,
        "total": total,
        "checkpoint_path": str(path),
    }


def save_csv(rows):
    shared_rows = []
    for row in rows:
        shared_rows.append(
            {
                "dataset": "MNIST",
                "model": "Direct-single QAT",
                "construction": "Train 10x784 complex matrix from scratch with QAT",
                "record_type": row["record_type"],
                "seed": row.get("seed", ""),
                "b_theta": row.get("b_theta", ""),
                "b_a": row.get("b_atten", ""),
                "test_accuracy_percent": row.get("test_acc_percent", ""),
                "mean_accuracy_percent": row.get("mean_test_acc_percent", ""),
                "sample_std_accuracy_percent": row.get("sample_std_test_acc_percent", ""),
                "best_validation_epoch": row.get("best_val_epoch", ""),
                "final_epoch": row.get("final_epoch", ""),
                "parameter_count_real_scalars": 2 * N4 * N0,
                "correct": row.get("correct", ""),
                "total": row.get("total", ""),
                "checkpoint_path": row.get("checkpoint_path", ""),
            }
        )
    write_shared_results(shared_rows)


def train_all():
    configure_cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    rows = []

    for seed in SEEDS:
        set_seed(seed)
        train_loader, val_loader, test_loader, train_indices, val_indices = (
            make_dataloaders(seed, device)
        )
        loaders = (train_loader, val_loader, test_loader)
        indices = (train_indices, val_indices)
        for b_theta in B_THETA_VALUES:
            for b_atten in B_ATTEN_VALUES:
                rows.append(
                    train_one(
                        seed, b_theta, b_atten, loaders, indices, device
                    )
                )
        del train_loader, val_loader, test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for b_theta in B_THETA_VALUES:
        for b_atten in B_ATTEN_VALUES:
            values = np.asarray(
                [
                    row["test_acc_percent"]
                    for row in rows
                    if row["b_theta"] == b_theta
                    and row["b_atten"] == b_atten
                ],
                dtype=np.float64,
            )
            rows.append(
                {
                    "record_type": "aggregate",
                    "seed": "",
                    "b_theta": b_theta,
                    "b_atten": b_atten,
                    "best_val_epoch": "",
                    "final_epoch": "",
                    "stopped_early": "",
                    "test_loss": "",
                    "test_acc_percent": "",
                    "mean_test_acc_percent": float(values.mean()),
                    "sample_std_test_acc_percent": float(values.std(ddof=1)),
                    "correct": "",
                    "total": "",
                    "checkpoint_path": "",
                }
            )

    save_csv(rows)
    print(f"Shared CSV updated: {SHARED_RESULTS_CSV.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="maximum epochs per run"
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

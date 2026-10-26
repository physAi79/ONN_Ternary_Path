"""Folded-single PTQ baseline for the four-stage full-precision MNIST model.

For every full-precision checkpoint, this script reconstructs the four
continuous complex propagation kernels and the three diagonal phase-bias
matrices, folds them into

    K_eff = K43 B3 K32 B2 K21 B1 K10,

normalizes K_eff by its maximum magnitude, and performs joint magnitude/phase
PTQ for (b_theta, b_atten) in {(1,1), (1,2), (2,1), (2,2)}.
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
from torch.utils.data import DataLoader, Subset
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
CHECKPOINT_PATH_TEMPLATE = str(
    OUTPUT_DIR / "main_optics_continuous_phase_alpha_model_seed_{seed}.pt"
)
CSV_SAVE_PATH = OUTPUT_DIR / "folded_single_ptq_three_seed_results.csv"

DATA_DIR = str(SOURCE_DIR / "data")
DOWNLOAD = True
BATCH_SIZE = 512
NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2
SEEDS = [42, 123, 2024]
B_THETA_VALUES = [1, 2]
B_ATTEN_VALUES = [1, 2]

WAVELENGTH = 532e-9
APERTURE_SIZE = 2e-3
DZ = 0.1
Z0, Z1, Z2, Z3, Z4 = 0.0, DZ, 2 * DZ, 3 * DZ, 4 * DZ
N0, N1, N2, N3, N4 = 28 * 28, 64, 128, 64, 10
EPS = 1e-12


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def raw_to_xy(raw_xy, aperture_size):
    return (aperture_size / 2.0) * torch.tanh(raw_xy)


def make_points(raw_xy, z, aperture_size):
    xy = raw_to_xy(raw_xy, aperture_size)
    z_column = torch.full(
        (xy.shape[0], 1), float(z), dtype=xy.dtype, device=xy.device
    )
    return torch.cat([xy, z_column], dim=1)


def continuous_kernel(points_from, points_to, theta, a_trans, wavelength):
    diff = points_to[:, None, :] - points_from[None, :, :]
    distance = torch.linalg.norm(diff, dim=2) + EPS
    propagation_phase = torch.remainder(
        (2.0 * math.pi / wavelength) * distance, 2.0 * math.pi
    )
    total_phase = propagation_phase + theta
    phase_factor = torch.complex(torch.cos(total_phase), torch.sin(total_phase))
    return (
        torch.clamp(a_trans, 0.0, 1.0) * phase_factor / distance
    ).to(torch.complex64)


def diagonal_phase(phase):
    factor = torch.complex(torch.cos(phase), torch.sin(phase)).to(torch.complex64)
    return torch.diag(factor)


def require_state(state, name):
    if name not in state:
        raise KeyError(f"Checkpoint is missing required tensor: {name}")
    return state[name]


def build_multistage_and_folded_matrices(checkpoint, device):
    state = {
        key: value.to(device)
        for key, value in checkpoint["model_state_dict"].items()
    }
    aperture_size = float(checkpoint.get("aperture_size", APERTURE_SIZE))
    wavelength = float(checkpoint.get("wavelength", WAVELENGTH))
    z_values = [
        float(checkpoint.get("z0", Z0)),
        float(checkpoint.get("z1", Z1)),
        float(checkpoint.get("z2", Z2)),
        float(checkpoint.get("z3", Z3)),
        float(checkpoint.get("z4", Z4)),
    ]

    point_names = ["raw_xy0", "raw_xy1", "raw_xy2", "raw_xy3", "raw_xy4"]
    points = [
        make_points(require_state(state, name), z, aperture_size)
        for name, z in zip(point_names, z_values)
    ]

    k10 = continuous_kernel(
        points[0],
        points[1],
        require_state(state, "path_theta10"),
        require_state(state, "path_a_trans10"),
        wavelength,
    )
    k21 = continuous_kernel(
        points[1],
        points[2],
        require_state(state, "path_theta21"),
        require_state(state, "path_a_trans21"),
        wavelength,
    )
    k32 = continuous_kernel(
        points[2],
        points[3],
        require_state(state, "path_theta32"),
        require_state(state, "path_a_trans32"),
        wavelength,
    )
    k43 = continuous_kernel(
        points[3],
        points[4],
        require_state(state, "path_theta43"),
        require_state(state, "path_a_trans43"),
        wavelength,
    )
    b1 = diagonal_phase(require_state(state, "bias_phase1"))
    b2 = diagonal_phase(require_state(state, "bias_phase2"))
    b3 = diagonal_phase(require_state(state, "bias_phase3"))

    k_eff = k43 @ b3 @ k32 @ b2 @ k21 @ b1 @ k10
    scale = torch.max(torch.abs(k_eff))
    if not torch.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Invalid folded-matrix scale: {scale.item()}")
    k_eff_normalized = k_eff / scale
    return (k10, k21, k32, k43), (b1, b2, b3), k_eff_normalized, scale


def quantize_folded_matrix(k_eff_normalized, b_theta, b_atten):
    theta_levels = 2 ** int(b_theta)
    atten_levels = 2 ** int(b_atten)
    delta_theta = 2.0 * math.pi / theta_levels
    delta_atten = 1.0 / (atten_levels - 1)

    magnitude = torch.clamp(torch.abs(k_eff_normalized), 0.0, 1.0)
    magnitude_q = torch.clamp(
        delta_atten * torch.round(magnitude / delta_atten), 0.0, 1.0
    )

    phase = torch.remainder(torch.angle(k_eff_normalized), 2.0 * math.pi)
    phase_index = torch.remainder(
        torch.round(phase / delta_theta), theta_levels
    )
    phase_q = delta_theta * phase_index
    phase_factor = torch.complex(torch.cos(phase_q), torch.sin(phase_q))
    return (magnitude_q * phase_factor).to(torch.complex64)


def encode_input_phase(x):
    phase = math.pi * x.view(x.size(0), -1).to(torch.float32)
    return torch.complex(torch.cos(phase), torch.sin(phase)).to(torch.complex64)


def normalize_field(psi):
    power = torch.sum(torch.abs(psi) ** 2, dim=1, keepdim=True)
    return psi / torch.sqrt(power + EPS)


def multistage_output(psi0, kernels, biases):
    k10, k21, k32, k43 = kernels
    b1, b2, b3 = biases
    psi1 = psi0 @ k10.T @ b1
    psi2 = psi1 @ k21.T @ b2
    psi3 = psi2 @ k32.T @ b3
    return normalize_field(psi3 @ k43.T)


class FixedSingleMatrixModel(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.n4 = matrix.shape[0]
        self.register_buffer("matrix", matrix)

    def complex_output(self, x):
        return normalize_field(encode_input_phase(x) @ self.matrix.T)

    def forward(self, x):
        intensity = torch.abs(self.complex_output(x)) ** 2
        return torch.log(intensity + EPS)


def make_loader(dataset, device):
    kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY and device.type == "cuda",
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(dataset, **kwargs)


def make_test_loader(checkpoint, device):
    data_dir = checkpoint.get("data_dir", DATA_DIR)
    test_set = datasets.MNIST(
        root=Path(data_dir),
        train=False,
        transform=transforms.ToTensor(),
        download=DOWNLOAD,
    )
    test_indices = checkpoint.get("test_indices")
    if test_indices is None:
        manifest = checkpoint.get("dataset_split_manifest")
        if isinstance(manifest, dict):
            test_indices = manifest.get("test_indices")
    if test_indices is not None:
        if torch.is_tensor(test_indices):
            test_indices = test_indices.cpu().tolist()
        test_set = Subset(test_set, test_indices)
    return make_loader(test_set, device)


def evaluate(model, loader, device):
    criterion = nn.KLDivLoss(reduction="batchmean")
    total_loss = 0.0
    total_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            targets = F.one_hot(labels, num_classes=N4).to(torch.float32)
            loss = criterion(F.log_softmax(logits, dim=1), targets)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += batch_size
    return total_loss / total, total_correct / total, total_correct, total


def verify_folding(loader, kernels, biases, folded_matrix, device):
    max_relative_error = 0.0
    rms_numerator = 0.0
    rms_denominator = 0.0
    prediction_mismatches = 0
    total = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            psi0 = encode_input_phase(images)
            y_multi = multistage_output(psi0, kernels, biases)
            y_folded = normalize_field(psi0 @ folded_matrix.T)

            difference = y_multi - y_folded
            per_sample_relative = torch.linalg.vector_norm(
                difference, dim=1
            ) / torch.clamp(
                torch.linalg.vector_norm(y_multi, dim=1), min=EPS
            )
            max_relative_error = max(
                max_relative_error, float(per_sample_relative.max().item())
            )
            rms_numerator += float(torch.sum(torch.abs(difference) ** 2).item())
            rms_denominator += float(torch.sum(torch.abs(y_multi) ** 2).item())
            prediction_mismatches += int(
                (
                    torch.argmax(torch.abs(y_multi) ** 2, dim=1)
                    != torch.argmax(torch.abs(y_folded) ** 2, dim=1)
                )
                .sum()
                .item()
            )
            total += images.size(0)

    relative_frobenius_error = math.sqrt(
        rms_numerator / max(rms_denominator, EPS)
    )
    return max_relative_error, relative_frobenius_error, prediction_mismatches, total


def save_csv(rows):
    shared_rows = []
    for row in rows:
        if row["model"] != "folded_single_ptq":
            continue
        shared_rows.append(
            {
                "dataset": "MNIST",
                "model": "Folded-single PTQ",
                "construction": "Quantize normalized K_eff",
                "record_type": row["record_type"],
                "seed": row.get("seed", ""),
                "b_theta": row.get("b_theta", ""),
                "b_a": row.get("b_atten", ""),
                "test_accuracy_percent": row.get("test_acc_percent", ""),
                "mean_accuracy_percent": row.get("mean_test_acc_percent", ""),
                "sample_std_accuracy_percent": row.get("sample_std_test_acc_percent", ""),
                "parameter_count_real_scalars": 2 * N4 * N0,
                "fold_scale": row.get("fold_scale", ""),
                "fold_max_relative_error": row.get("fold_max_relative_error", ""),
                "fold_relative_frobenius_error": row.get("fold_relative_frobenius_error", ""),
                "fold_prediction_mismatches": row.get("fold_prediction_mismatches", ""),
                "correct": row.get("correct", ""),
                "total": row.get("total", ""),
                "checkpoint_path": row.get("checkpoint_path", ""),
            }
        )
    write_shared_results(shared_rows)


def run(checkpoint_template):
    configure_cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    rows = []

    for seed in SEEDS:
        set_seed(seed)
        checkpoint_path = Path(checkpoint_template.format(seed=seed))
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint for seed {seed} not found: {checkpoint_path.resolve()}"
            )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        kernels, biases, folded_fp, fold_scale = (
            build_multistage_and_folded_matrices(checkpoint, device)
        )
        test_loader = make_test_loader(checkpoint, device)

        max_error, fro_error, mismatches, verify_total = verify_folding(
            test_loader, kernels, biases, folded_fp, device
        )
        print("\n" + "=" * 90)
        print(f"Seed {seed}: {checkpoint_path.resolve()}")
        print(
            "Fold verification | "
            f"max relative error={max_error:.6e} | "
            f"relative Frobenius error={fro_error:.6e} | "
            f"prediction mismatches={mismatches}/{verify_total}"
        )

        fp_model = FixedSingleMatrixModel(folded_fp).to(device)
        fp_loss, fp_acc, fp_correct, fp_total = evaluate(
            fp_model, test_loader, device
        )
        rows.append(
            {
                "record_type": "seed_result",
                "seed": seed,
                "model": "folded_single_fp",
                "b_theta": "FP",
                "b_atten": "FP",
                "test_loss": fp_loss,
                "test_acc_percent": 100.0 * fp_acc,
                "mean_test_acc_percent": "",
                "sample_std_test_acc_percent": "",
                "correct": fp_correct,
                "total": fp_total,
                "fold_scale": float(fold_scale.item()),
                "fold_max_relative_error": max_error,
                "fold_relative_frobenius_error": fro_error,
                "fold_prediction_mismatches": mismatches,
                "checkpoint_path": str(checkpoint_path),
            }
        )
        print(f"Folded-single FP accuracy: {100*fp_acc:.6f}%")

        for b_theta in B_THETA_VALUES:
            for b_atten in B_ATTEN_VALUES:
                matrix_q = quantize_folded_matrix(
                    folded_fp, b_theta, b_atten
                )
                model = FixedSingleMatrixModel(matrix_q).to(device)
                loss, acc, correct, total = evaluate(model, test_loader, device)
                rows.append(
                    {
                        "record_type": "seed_result",
                        "seed": seed,
                        "model": "folded_single_ptq",
                        "b_theta": b_theta,
                        "b_atten": b_atten,
                        "test_loss": loss,
                        "test_acc_percent": 100.0 * acc,
                        "mean_test_acc_percent": "",
                        "sample_std_test_acc_percent": "",
                        "correct": correct,
                        "total": total,
                        "fold_scale": float(fold_scale.item()),
                        "fold_max_relative_error": max_error,
                        "fold_relative_frobenius_error": fro_error,
                        "fold_prediction_mismatches": mismatches,
                        "checkpoint_path": str(checkpoint_path),
                    }
                )
                print(
                    f"Folded-single PTQ ({b_theta},{b_atten}) accuracy: "
                    f"{100*acc:.6f}%"
                )

        del test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    groups = [("folded_single_fp", "FP", "FP")] + [
        ("folded_single_ptq", b_theta, b_atten)
        for b_theta in B_THETA_VALUES
        for b_atten in B_ATTEN_VALUES
    ]
    for model_name, b_theta, b_atten in groups:
        values = np.asarray(
            [
                row["test_acc_percent"]
                for row in rows
                if row["model"] == model_name
                and row["b_theta"] == b_theta
                and row["b_atten"] == b_atten
            ],
            dtype=np.float64,
        )
        matching_seed_rows = [
            row for row in rows
            if row["record_type"] == "seed_result"
            and row["model"] == model_name
            and row["b_theta"] == b_theta
            and row["b_atten"] == b_atten
        ]
        rows.append(
            {
                "record_type": "aggregate",
                "seed": "",
                "model": model_name,
                "b_theta": b_theta,
                "b_atten": b_atten,
                "test_loss": "",
                "test_acc_percent": "",
                "mean_test_acc_percent": float(values.mean()),
                "sample_std_test_acc_percent": float(values.std(ddof=1)),
                "correct": "",
                "total": "",
                "fold_scale": "",
                "fold_max_relative_error": max(row["fold_max_relative_error"] for row in matching_seed_rows),
                "fold_relative_frobenius_error": max(row["fold_relative_frobenius_error"] for row in matching_seed_rows),
                "fold_prediction_mismatches": sum(row["fold_prediction_mismatches"] for row in matching_seed_rows),
                "checkpoint_path": "",
            }
        )

    save_csv(rows)
    print(f"Shared CSV updated: {SHARED_RESULTS_CSV.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-template",
        default=CHECKPOINT_PATH_TEMPLATE,
        help="checkpoint path template containing {seed}",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="test batch size"
    )
    return parser.parse_args()


def main():
    global BATCH_SIZE
    args = parse_args()
    BATCH_SIZE = args.batch_size
    run(args.checkpoint_template)


if __name__ == "__main__":
    main()

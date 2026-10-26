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


# ============================================================
# Fixed running configuration
# ============================================================
CHECKPOINT_PATH_TEMPLATE = "main_optics_continuous_phase_alpha_model_seed_{seed}.pt"
CSV_SAVE_PATH = Path("continuous_alpha_phase_quantized_inference_acc.csv")

DATA_DIR = "./data"
DOWNLOAD = True

BATCH_SIZE = 512
NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2

SEEDS = [42, 123, 2024]

B_THETA_VALUES = [1] #, 2, 3, 6, 8, 16]

# alpha keeps continuous. This value is only a placeholder
# because the model helper stores both b_theta and b_alpha.
B_ALPHA_CONTINUOUS_PLACEHOLDER = 16


# ============================================================
# Default optical configuration
# Checkpoint metadata will override these when available.
# ============================================================
WAVELENGTH = 532e-9
APERTURE_SIZE = 2e-3

dz = 0.1
Z0 = 0.0
Z1 = dz
Z2 = 2 * dz
Z3 = 3 * dz
Z4 = 4 * dz

N0 = 28 * 28
N1 = 64
N2 = 128
N3 = 64
N4 = 10

USE_LOG_INTENSITY = True
EPS = 1e-12

A_TRANS_INIT_VALUE = 1.0
A_TRANS_INIT_STD = 1e-3


class OpticalPathModel(nn.Module):
    def __init__(
        self,
        n0=N0,
        n1=N1,
        n2=N2,
        n3=N3,
        n4=N4,
        wavelength=WAVELENGTH,
        z0=Z0,
        z1=Z1,
        z2=Z2,
        z3=Z3,
        z4=Z4,
        aperture_size=APERTURE_SIZE,
        use_log_intensity=USE_LOG_INTENSITY,
        b_theta=16,
        b_alpha=16,
        quantize_phase_enabled=True,
        quantize_alpha_enabled=False,
    ):
        super().__init__()

        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4

        self.wavelength = wavelength
        self.k = 2.0 * math.pi / wavelength

        self.aperture_size = aperture_size
        self.half_aperture = aperture_size / 2.0
        self.use_log_intensity = use_log_intensity

        self.quantize_phase_enabled = quantize_phase_enabled
        self.quantize_alpha_enabled = quantize_alpha_enabled

        self.set_quantization_bits(b_theta, b_alpha)

        xy0 = self.make_input_line_xy(28, 28)
        xy1 = self.make_line_xy(n1)
        xy2 = self.make_line_xy(n2)
        xy3 = self.make_line_xy(n3)
        xy4 = self.make_line_xy(n4)

        self.raw_xy0 = nn.Parameter(self.xy_to_raw(xy0), requires_grad=False)
        self.raw_xy1 = nn.Parameter(self.xy_to_raw(xy1), requires_grad=False)
        self.raw_xy2 = nn.Parameter(self.xy_to_raw(xy2), requires_grad=False)
        self.raw_xy3 = nn.Parameter(self.xy_to_raw(xy3), requires_grad=False)
        self.raw_xy4 = nn.Parameter(self.xy_to_raw(xy4), requires_grad=False)

        self.register_buffer("z0_tensor", torch.full((n0, 1), z0, dtype=torch.float32))
        self.register_buffer("z1_tensor", torch.full((n1, 1), z1, dtype=torch.float32))
        self.register_buffer("z2_tensor", torch.full((n2, 1), z2, dtype=torch.float32))
        self.register_buffer("z3_tensor", torch.full((n3, 1), z3, dtype=torch.float32))
        self.register_buffer("z4_tensor", torch.full((n4, 1), z4, dtype=torch.float32))

        self.path_theta10 = nn.Parameter(torch.zeros(n1, n0))
        self.path_theta21 = nn.Parameter(torch.zeros(n2, n1))
        self.path_theta32 = nn.Parameter(torch.zeros(n3, n2))
        self.path_theta43 = nn.Parameter(torch.zeros(n4, n3))

        self.path_a_trans10 = nn.Parameter(
            torch.clamp(
                A_TRANS_INIT_VALUE + A_TRANS_INIT_STD * torch.randn(n1, n0),
                0.0,
                1.0,
            )
        )
        self.path_a_trans21 = nn.Parameter(
            torch.clamp(
                A_TRANS_INIT_VALUE + A_TRANS_INIT_STD * torch.randn(n2, n1),
                0.0,
                1.0,
            )
        )
        self.path_a_trans32 = nn.Parameter(
            torch.clamp(
                A_TRANS_INIT_VALUE + A_TRANS_INIT_STD * torch.randn(n3, n2),
                0.0,
                1.0,
            )
        )
        self.path_a_trans43 = nn.Parameter(
            torch.clamp(
                A_TRANS_INIT_VALUE + A_TRANS_INIT_STD * torch.randn(n4, n3),
                0.0,
                1.0,
            )
        )

        self.bias_phase1 = nn.Parameter(torch.zeros(n1))
        self.bias_phase2 = nn.Parameter(torch.zeros(n2))
        self.bias_phase3 = nn.Parameter(torch.zeros(n3))

    def set_quantization_bits(self, b_theta, b_alpha):
        self.b_theta = int(b_theta)
        self.theta_levels = 2 ** self.b_theta
        self.delta_theta = 2.0 * math.pi / self.theta_levels

        self.b_alpha = int(b_alpha)
        self.alpha_levels = 2 ** self.b_alpha
        self.delta_alpha = 1.0 / (self.alpha_levels - 1)

    def make_input_line_xy(self, nx, ny):
        return self.make_line_xy(nx * ny)

    def make_line_xy(self, n):
        xs = torch.linspace(
            -0.95 * self.half_aperture,
            0.95 * self.half_aperture,
            n,
            dtype=torch.float32,
        )
        ys = torch.zeros_like(xs)
        return torch.stack([xs, ys], dim=1)

    def xy_to_raw(self, xy):
        normalized = torch.clamp(
            xy / self.half_aperture,
            min=-0.999,
            max=0.999,
        )
        return torch.atanh(normalized)

    def raw_to_xy(self, raw_xy):
        return self.half_aperture * torch.tanh(raw_xy)

    def get_xy(self):
        xy0 = self.raw_to_xy(self.raw_xy0)
        xy1 = self.raw_to_xy(self.raw_xy1)
        xy2 = self.raw_to_xy(self.raw_xy2)
        xy3 = self.raw_to_xy(self.raw_xy3)
        xy4 = self.raw_to_xy(self.raw_xy4)
        return xy0, xy1, xy2, xy3, xy4

    def make_points(self, xy, z_tensor):
        return torch.cat([xy, z_tensor], dim=1)

    def quantize_phase_value(self, theta):
        theta_mod = torch.remainder(theta, 2.0 * math.pi)
        theta_index = torch.round(theta_mod / self.delta_theta)
        theta_index = torch.remainder(theta_index, self.theta_levels)
        theta_q = self.delta_theta * theta_index
        return theta_q

    def quantize_alpha_value(self, alpha):
        alpha_clamped = torch.clamp(alpha, 0.0, 1.0)
        alpha_q = self.delta_alpha * torch.round(alpha_clamped / self.delta_alpha)
        alpha_q = torch.clamp(alpha_q, 0.0, 1.0)
        return alpha_q

    def apply_phase_bias(self, psi, bias_phase):
        if self.quantize_phase_enabled:
            bias_phase_used = self.quantize_phase_value(bias_phase)
        else:
            bias_phase_used = bias_phase

        bias_factor = torch.complex(
            torch.cos(bias_phase_used),
            torch.sin(bias_phase_used),
        ).to(torch.complex64)

        return psi * bias_factor[None, :]

    def propagation_kernel(self, points_from, points_to, path_theta, path_alpha):
        diff = points_to[:, None, :] - points_from[None, :, :]
        r = torch.linalg.norm(diff, dim=2) + EPS

        propagation_phase = torch.remainder(self.k * r, 2.0 * math.pi)

        base_kernel = torch.complex(
            torch.cos(propagation_phase),
            torch.sin(propagation_phase),
        ) / r

        if self.quantize_alpha_enabled:
            alpha_used = self.quantize_alpha_value(path_alpha)
        else:
            alpha_used = torch.clamp(path_alpha, 0.0, 1.0)

        if self.quantize_phase_enabled:
            theta_used = self.quantize_phase_value(path_theta)
        else:
            theta_used = path_theta

        path_phase_factor = torch.complex(
            torch.cos(theta_used),
            torch.sin(theta_used),
        )

        kernel = base_kernel * alpha_used * path_phase_factor

        return kernel.to(torch.complex64)

    def normalize_field(self, psi):
        intensity = torch.abs(psi) ** 2
        total_energy = torch.sum(intensity, dim=1, keepdim=True)
        return psi / torch.sqrt(total_energy + EPS)

    def encode_input_phase(self, x):
        phase = math.pi * x.view(x.size(0), -1).to(torch.float32)
        psi = torch.complex(torch.cos(phase), torch.sin(phase))
        return psi.to(torch.complex64)

    def forward(self, x):
        xy0, xy1, xy2, xy3, xy4 = self.get_xy()

        holes0 = self.make_points(xy0, self.z0_tensor)
        holes1 = self.make_points(xy1, self.z1_tensor)
        holes2 = self.make_points(xy2, self.z2_tensor)
        holes3 = self.make_points(xy3, self.z3_tensor)
        holes4 = self.make_points(xy4, self.z4_tensor)

        k10 = self.propagation_kernel(
            holes0,
            holes1,
            self.path_theta10,
            self.path_a_trans10,
        )
        k21 = self.propagation_kernel(
            holes1,
            holes2,
            self.path_theta21,
            self.path_a_trans21,
        )
        k32 = self.propagation_kernel(
            holes2,
            holes3,
            self.path_theta32,
            self.path_a_trans32,
        )
        k43 = self.propagation_kernel(
            holes3,
            holes4,
            self.path_theta43,
            self.path_a_trans43,
        )

        psi0 = self.encode_input_phase(x)

        psi1 = psi0 @ k10.T
        psi1 = self.apply_phase_bias(psi1, self.bias_phase1)
        psi1 = self.normalize_field(psi1)

        psi2 = psi1 @ k21.T
        psi2 = self.apply_phase_bias(psi2, self.bias_phase2)
        psi2 = self.normalize_field(psi2)

        psi3 = psi2 @ k32.T
        psi3 = self.apply_phase_bias(psi3, self.bias_phase3)
        psi3 = self.normalize_field(psi3)

        psi4 = psi3 @ k43.T
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


def clamp_a_trans_parameters(model):
    with torch.no_grad():
        model.path_a_trans10.clamp_(0.0, 1.0)
        model.path_a_trans21.clamp_(0.0, 1.0)
        model.path_a_trans32.clamp_(0.0, 1.0)
        model.path_a_trans43.clamp_(0.0, 1.0)


def make_transform():
    return transforms.Compose([transforms.ToTensor()])


def make_loader(dataset, batch_size, shuffle, device):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY and device.type == "cuda",
    }

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    return DataLoader(dataset, **loader_kwargs)


def make_test_loader_from_checkpoint(checkpoint, device, batch_size=BATCH_SIZE):
    data_dir = checkpoint.get("data_dir", DATA_DIR)

    transform = make_transform()

    test_set = datasets.MNIST(
        root=Path(data_dir),
        train=False,
        transform=transform,
        download=DOWNLOAD,
    )

    test_indices = checkpoint.get("test_indices", None)

    if test_indices is None:
        split_manifest = checkpoint.get("dataset_split_manifest", None)
        if split_manifest is not None:
            test_indices = split_manifest.get("test_indices", None)

    if test_indices is not None:
        if torch.is_tensor(test_indices):
            test_indices = test_indices.cpu().tolist()
        test_set = Subset(test_set, test_indices)

    return make_loader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )


def make_target_probs(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).to(torch.float32)


def evaluate_with_count(model, loader, criterion, device):
    model.eval()

    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)

            log_probs = F.log_softmax(logits, dim=1)
            target_probs = make_target_probs(labels, model.n4).to(device)

            loss = criterion(log_probs, target_probs)

            preds = logits.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += loss.detach() * batch_size
            total_correct += (preds == labels).sum()
            total += batch_size

    correct = int(total_correct.item())
    avg_loss = (total_loss / total).item()
    acc = correct / total

    return avg_loss, acc, correct, total


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = OpticalPathModel(
        n0=checkpoint.get("n0", N0),
        n1=checkpoint.get("n1", N1),
        n2=checkpoint.get("n2", N2),
        n3=checkpoint.get("n3", N3),
        n4=checkpoint.get("n4", N4),
        wavelength=checkpoint.get("wavelength", WAVELENGTH),
        z0=checkpoint.get("z0", Z0),
        z1=checkpoint.get("z1", Z1),
        z2=checkpoint.get("z2", Z2),
        z3=checkpoint.get("z3", Z3),
        z4=checkpoint.get("z4", Z4),
        aperture_size=checkpoint.get("aperture_size", APERTURE_SIZE),
        use_log_intensity=checkpoint.get("use_log_intensity", USE_LOG_INTENSITY),
        b_theta=16,
        b_alpha=B_ALPHA_CONTINUOUS_PLACEHOLDER,
        quantize_phase_enabled=True,
        quantize_alpha_enabled=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    clamp_a_trans_parameters(model)
    model.eval()

    return model, checkpoint


def save_results_to_csv(rows, csv_save_path):
    fieldnames = [
        "record_type",
        "seed",
        "b_theta",
        "test_acc",
        "test_acc_percent",
        "mean_test_acc",
        "mean_test_acc_percent",
        "sample_std_test_acc",
        "sample_std_test_acc_percent",
        "test_loss",
        "correct",
        "total",
        "theta_inference",
        "bias_phase_inference",
        "alpha_inference",
        "checkpoint_path",
    ]

    with csv_save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_phase_quantized_continuous_alpha_inference_sweep():
    configure_cuda()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available. Running on CPU.")

    criterion = nn.KLDivLoss(reduction="batchmean")
    rows = []
    seed_results_by_bit = {b_theta: [] for b_theta in B_THETA_VALUES}

    print("\nPhase-quantized / continuous-alpha inference sweep")
    print("theta: quantized")
    print("bias_phase: quantized")
    print("alpha: continuous")

    for seed in SEEDS:
        set_seed(seed)
        checkpoint_path = Path(CHECKPOINT_PATH_TEMPLATE.format(seed=seed))

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint for seed {seed} was not found: {checkpoint_path.resolve()}"
            )

        print("\n" + "=" * 90)
        print(f"Seed {seed}: loading checkpoint {checkpoint_path.resolve()}")
        print("b_theta | test acc (%) | test loss | correct / total")

        model, checkpoint = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        test_loader = make_test_loader_from_checkpoint(
            checkpoint=checkpoint,
            device=device,
            batch_size=BATCH_SIZE,
        )

        for b_theta in B_THETA_VALUES:
            model.set_quantization_bits(
                b_theta=b_theta,
                b_alpha=B_ALPHA_CONTINUOUS_PLACEHOLDER,
            )

            test_loss, test_acc, correct, total = evaluate_with_count(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
            )

            seed_results_by_bit[b_theta].append(test_acc)
            rows.append(
                {
                    "record_type": "seed_result",
                    "seed": seed,
                    "b_theta": b_theta,
                    "test_acc": test_acc,
                    "test_acc_percent": test_acc * 100.0,
                    "mean_test_acc": "",
                    "mean_test_acc_percent": "",
                    "sample_std_test_acc": "",
                    "sample_std_test_acc_percent": "",
                    "test_loss": test_loss,
                    "correct": correct,
                    "total": total,
                    "theta_inference": "quantized",
                    "bias_phase_inference": "quantized",
                    "alpha_inference": "continuous",
                    "checkpoint_path": str(checkpoint_path),
                }
            )

            print(
                f"{b_theta:7d} | "
                f"{test_acc * 100:12.2f} | "
                f"{test_loss:9.4f} | "
                f"{correct:7d} / {total:<7d}"
            )

        del model, test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("Three-seed accuracy statistics")
    print("b_theta | mean acc (%) | sample std (%)")

    for b_theta in B_THETA_VALUES:
        accuracies = np.asarray(seed_results_by_bit[b_theta], dtype=np.float64)
        mean_test_acc = float(np.mean(accuracies))
        sample_std_test_acc = float(np.std(accuracies, ddof=1))
        mean_test_acc_percent = mean_test_acc * 100.0
        sample_std_test_acc_percent = sample_std_test_acc * 100.0

        rows.append(
            {
                "record_type": "aggregate",
                "seed": "",
                "b_theta": b_theta,
                "test_acc": "",
                "test_acc_percent": "",
                "mean_test_acc": mean_test_acc,
                "mean_test_acc_percent": mean_test_acc_percent,
                "sample_std_test_acc": sample_std_test_acc,
                "sample_std_test_acc_percent": sample_std_test_acc_percent,
                "test_loss": "",
                "correct": "",
                "total": "",
                "theta_inference": "quantized",
                "bias_phase_inference": "quantized",
                "alpha_inference": "continuous",
                "checkpoint_path": "",
            }
        )

        print(
            f"{b_theta:7d} | "
            f"{mean_test_acc_percent:12.6f} | "
            f"{sample_std_test_acc_percent:14.8f}"
        )

    save_results_to_csv(rows, CSV_SAVE_PATH)

    print(f"\nSaved CSV to: {CSV_SAVE_PATH.resolve()}")


def main():
    run_phase_quantized_continuous_alpha_inference_sweep()


if __name__ == "__main__":
    main()

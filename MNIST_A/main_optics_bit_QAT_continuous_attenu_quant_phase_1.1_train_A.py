import csv
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


DATA_DIR = "./data"
EPOCHS = 100
BATCH_SIZE = 512
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA_LOSS = 1e-4
EARLY_STOPPING_MIN_DELTA_ACC = 1e-4
LR = 1e-2
LR_STEP_SIZE = 10
LR_GAMMA = 0.99
WEIGHT_DECAY = 0.0
SEEDS = [42, 123, 2024]
SEED = SEEDS[0]
VAL_RATIO = 0.1
DOWNLOAD = True

NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2

WAVELENGTH = 532e-9
APERTURE_SIZE = 2e-3

DZ = 0.1
Z0 = 0.0
Z1 = DZ
Z2 = 2 * DZ
Z3 = 3 * DZ
Z4 = 4 * DZ

N0 = 28 * 28
N1 = 64
N2 = 128
N3 = 64
N4 = 10

USE_LOG_INTENSITY = True
EPS = 1e-12

QAT_QUANTIZE_PHASE = True
QAT_QUANTIZE_TRANSMITTANCE = False

B_THETA_VALUES = [1] #, 2, 3, 6, 8, 16]

A_TRANS_INIT_VALUE = 1.0
A_TRANS_INIT_STD = 1e-3

SPLIT_MANIFEST_PATH = Path(
    "main_optics_theta_QAT_attenuation_continuous_dataset_split_manifest.pt"
)

SUMMARY_PATH = Path(
    "main_optics_theta_QAT_attenuation_continuous_summary.pt"
)

CSV_SAVE_PATH = Path(
    "main_optics_theta_QAT_attenuation_continuous_three_seed_results.csv"
)

MODE = "train"
INFER_CHECKPOINT_PATH = None


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
        b_theta=6,
        qat_quantize_phase=QAT_QUANTIZE_PHASE,
        qat_quantize_transmittance=QAT_QUANTIZE_TRANSMITTANCE,
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

        self.qat_quantize_phase = qat_quantize_phase
        self.qat_quantize_transmittance = qat_quantize_transmittance

        self.b_theta = b_theta
        self.theta_levels = 2 ** b_theta
        self.delta_theta = 2.0 * math.pi / self.theta_levels

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

    def quantize_phase_ste(self, theta):
        theta_mod = torch.remainder(theta, 2.0 * math.pi)
        theta_index = torch.round(theta_mod / self.delta_theta)
        theta_index = torch.remainder(theta_index, self.theta_levels)
        theta_q = self.delta_theta * theta_index
        return theta + (theta_q - theta).detach()

    def apply_phase_bias(self, psi, bias_phase):
        if self.qat_quantize_phase:
            bias_phase_used = self.quantize_phase_ste(bias_phase)
        else:
            bias_phase_used = bias_phase

        bias_factor = torch.complex(
            torch.cos(bias_phase_used),
            torch.sin(bias_phase_used),
        ).to(torch.complex64)

        return psi * bias_factor[None, :]

    def propagation_kernel(self, points_from, points_to, path_theta, path_a_trans):
        diff = points_to[:, None, :] - points_from[None, :, :]
        r = torch.linalg.norm(diff, dim=2) + EPS

        propagation_phase = torch.remainder(self.k * r, 2.0 * math.pi)

        base_kernel = torch.complex(
            torch.cos(propagation_phase),
            torch.sin(propagation_phase),
        ) / r

        a_trans_used = torch.clamp(path_a_trans, 0.0, 1.0)

        if self.qat_quantize_phase:
            path_theta_used = self.quantize_phase_ste(path_theta)
        else:
            path_theta_used = path_theta

        path_phase_factor = torch.complex(
            torch.cos(path_theta_used),
            torch.sin(path_theta_used),
        )

        kernel = base_kernel * a_trans_used * path_phase_factor
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


def make_transform():
    return transforms.Compose([transforms.ToTensor()])


def make_datasets_and_save_split_manifest(device):
    transform = make_transform()
    data_dir = Path(DATA_DIR)

    train_full = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=DOWNLOAD,
    )

    test_set = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=DOWNLOAD,
    )

    val_size = int(len(train_full) * VAL_RATIO)
    train_size = len(train_full) - val_size

    generator = torch.Generator().manual_seed(SEED)

    train_set, val_set = random_split(
        train_full,
        [train_size, val_size],
        generator=generator,
    )

    train_indices = torch.as_tensor(train_set.indices, dtype=torch.long)
    val_indices = torch.as_tensor(val_set.indices, dtype=torch.long)
    test_indices = torch.arange(len(test_set), dtype=torch.long)

    split_manifest = {
        "dataset": "MNIST",
        "data_dir": str(Path(DATA_DIR).resolve()),
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "train_full_size": len(train_full),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "test_set_rule": "torchvision.datasets.MNIST(train=False), official fixed MNIST test set",
        "transform": "transforms.ToTensor()",
    }

    torch.save(split_manifest, SPLIT_MANIFEST_PATH)

    train_loader = make_loader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        device=device,
    )

    val_loader = make_loader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        device=device,
    )

    test_loader = make_loader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        device=device,
    )

    print(f"Saved dataset split manifest to: {SPLIT_MANIFEST_PATH.resolve()}")
    print(f"Train size: {len(train_set)}")
    print(f"Val size: {len(val_set)}")
    print(f"Test size: {len(test_set)}")

    return train_loader, val_loader, test_loader


def make_test_loader_for_inference(device, batch_size=BATCH_SIZE):
    transform = make_transform()

    test_set = datasets.MNIST(
        root=Path(DATA_DIR),
        train=False,
        transform=transform,
        download=DOWNLOAD,
    )

    if SPLIT_MANIFEST_PATH.exists():
        manifest = torch.load(SPLIT_MANIFEST_PATH, map_location="cpu")
        test_indices = manifest.get("test_indices", None)

        if test_indices is not None:
            test_set = Subset(test_set, test_indices.tolist())

    return make_loader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )


def make_target_probs(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).to(torch.float32)


def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)

    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device)
    total_count = 0

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            log_probs = F.log_softmax(logits, dim=1)
            target_probs = make_target_probs(labels, N4).to(device)

            loss = criterion(log_probs, target_probs)

            if train:
                loss.backward()
                optimizer.step()
                clamp_a_trans_parameters(model)

            batch_size = labels.size(0)
            preds = logits.argmax(dim=1)

            total_loss += loss.detach() * batch_size
            total_correct += (preds == labels).sum()
            total_count += batch_size

    avg_loss = (total_loss / total_count).item()
    avg_acc = (total_correct / total_count).item()

    return avg_loss, avg_acc


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
            target_probs = make_target_probs(labels, N4).to(device)

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


def quantize_phase_for_save(theta, b_theta):
    levels = 2 ** b_theta
    delta_theta = 2.0 * math.pi / levels

    theta_mod = torch.remainder(theta, 2.0 * math.pi)
    theta_index = torch.round(theta_mod / delta_theta)
    theta_index = torch.remainder(theta_index, levels)
    theta_q = delta_theta * theta_index

    return theta_q


def save_hole_layout_visualization(xy_list, save_path):
    fig, axes = plt.subplots(1, len(xy_list), figsize=(4 * len(xy_list), 4))

    if len(xy_list) == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        xy = xy_list[idx].detach().cpu().numpy()

        ax.scatter(
            xy[:, 0] * 1e3,
            xy[:, 1] * 1e3,
            s=16,
            color="black",
        )
        ax.set_title(f"Layer {idx} line-array nodes")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_path_parameter_visualization(
    path_theta,
    path_a_trans,
    title,
    save_path,
    b_theta,
):
    path_theta_tensor = path_theta.detach().cpu()
    path_theta_q = quantize_phase_for_save(path_theta_tensor, b_theta)
    path_theta_q_np = path_theta_q.numpy()

    a_trans_continuous = torch.clamp(path_a_trans.detach().cpu(), 0.0, 1.0)
    a_trans_np = a_trans_continuous.numpy()

    phase = np.angle(np.exp(1j * path_theta_q_np))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(
        phase,
        aspect="auto",
        cmap="twilight",
        origin="lower",
    )
    axes[0].set_title(f"{title} quantized phase, b_theta={b_theta}")
    axes[0].set_xlabel("previous layer node i")
    axes[0].set_ylabel("next layer node j")
    fig.colorbar(im0, ax=axes[0], label="quantized phase (rad)")

    im1 = axes[1].imshow(
        a_trans_np,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(f"{title} continuous attenuation")
    axes[1].set_xlabel("previous layer node i")
    axes[1].set_ylabel("next layer node j")
    fig.colorbar(im1, ax=axes[1], label="continuous amplitude transmittance")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_trained_model(
    model,
    initial_xy,
    b_theta,
    expected_params,
    best_val_acc,
    best_val_epoch,
    final_epoch,
    training_converged,
    test_loss,
    test_acc,
    correct,
    total,
):
    xy0, xy1, xy2, xy3, xy4 = model.get_xy()
    initial_xy0, initial_xy1, initial_xy2, initial_xy3, initial_xy4 = initial_xy

    path_a_trans10_continuous = torch.clamp(model.path_a_trans10.detach().cpu(), 0.0, 1.0)
    path_a_trans21_continuous = torch.clamp(model.path_a_trans21.detach().cpu(), 0.0, 1.0)
    path_a_trans32_continuous = torch.clamp(model.path_a_trans32.detach().cpu(), 0.0, 1.0)
    path_a_trans43_continuous = torch.clamp(model.path_a_trans43.detach().cpu(), 0.0, 1.0)

    path_theta10_quantized = quantize_phase_for_save(model.path_theta10.detach().cpu(), b_theta)
    path_theta21_quantized = quantize_phase_for_save(model.path_theta21.detach().cpu(), b_theta)
    path_theta32_quantized = quantize_phase_for_save(model.path_theta32.detach().cpu(), b_theta)
    path_theta43_quantized = quantize_phase_for_save(model.path_theta43.detach().cpu(), b_theta)

    bias_phase1_quantized = quantize_phase_for_save(model.bias_phase1.detach().cpu(), b_theta)
    bias_phase2_quantized = quantize_phase_for_save(model.bias_phase2.detach().cpu(), b_theta)
    bias_phase3_quantized = quantize_phase_for_save(model.bias_phase3.detach().cpu(), b_theta)

    save_path = Path(
        f"main_optics_theta_QAT_attenuation_continuous_bt_{b_theta}_seed_{SEED}.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "xy0": xy0.detach().cpu(),
            "xy1": xy1.detach().cpu(),
            "xy2": xy2.detach().cpu(),
            "xy3": xy3.detach().cpu(),
            "xy4": xy4.detach().cpu(),
            "initial_xy0": initial_xy0,
            "initial_xy1": initial_xy1,
            "initial_xy2": initial_xy2,
            "initial_xy3": initial_xy3,
            "initial_xy4": initial_xy4,
            "path_theta10": model.path_theta10.detach().cpu(),
            "path_theta21": model.path_theta21.detach().cpu(),
            "path_theta32": model.path_theta32.detach().cpu(),
            "path_theta43": model.path_theta43.detach().cpu(),
            "path_theta10_quantized": path_theta10_quantized,
            "path_theta21_quantized": path_theta21_quantized,
            "path_theta32_quantized": path_theta32_quantized,
            "path_theta43_quantized": path_theta43_quantized,
            "path_a_trans10": model.path_a_trans10.detach().cpu(),
            "path_a_trans21": model.path_a_trans21.detach().cpu(),
            "path_a_trans32": model.path_a_trans32.detach().cpu(),
            "path_a_trans43": model.path_a_trans43.detach().cpu(),
            "path_a_trans10_continuous": path_a_trans10_continuous,
            "path_a_trans21_continuous": path_a_trans21_continuous,
            "path_a_trans32_continuous": path_a_trans32_continuous,
            "path_a_trans43_continuous": path_a_trans43_continuous,
            "bias_phase1": model.bias_phase1.detach().cpu(),
            "bias_phase2": model.bias_phase2.detach().cpu(),
            "bias_phase3": model.bias_phase3.detach().cpu(),
            "bias_phase1_quantized": bias_phase1_quantized,
            "bias_phase2_quantized": bias_phase2_quantized,
            "bias_phase3_quantized": bias_phase3_quantized,
            "wavelength": WAVELENGTH,
            "z0": Z0,
            "z1": Z1,
            "z2": Z2,
            "z3": Z3,
            "z4": Z4,
            "aperture_size": APERTURE_SIZE,
            "optical_geometry": (
                "1D line-array optical path model; "
                "MNIST image is flattened to 784 input line-array nodes"
            ),
            "input_encoding": "psi0 = exp(i * pi * image)",
            "experiment_type": (
                "QAT bit sweep for phase only; b_theta controls inter-layer path phase "
                "and output-node bias phase; attenuation is continuous"
            ),
            "qat_quantize_phase": QAT_QUANTIZE_PHASE,
            "qat_quantize_transmittance": QAT_QUANTIZE_TRANSMITTANCE,
            "attenuation_continuous": True,
            "b_theta": b_theta,
            "delta_theta": 2.0 * math.pi / (2 ** b_theta),
            "phase_quantization": (
                "Q_theta(x) = Delta_theta * "
                "(round((x mod 2*pi) / Delta_theta) mod 2^b_theta); "
                "used for both inter-layer path phase theta[j,i] and bias_phase_l[j]"
            ),
            "transmittance_quantization": "disabled; a_trans is continuous and clamped to [0, 1]",
            "bias_phase_quantization": (
                "bias_phase_l[j] uses the same Q_theta quantizer and the same b_theta "
                "as inter-layer transmission phase"
            ),
            "ste_backward": (
                "Straight-through estimator is used for path phase and bias phase only; "
                "attenuation is trained continuously"
            ),
            "propagation_kernel": (
                "K[j,i] = exp(i*k*r[j,i]) / r[j,i] "
                "* clamp(a_trans[j,i], 0, 1) "
                "* exp(i*Q_theta(path_theta[j,i]))"
            ),
            "output_phase_bias_formula": (
                "psi_l[j] = exp(i * Q_theta(bias_phase_l[j])) "
                "* sum_i K_l[j,i] * psi_{l-1}[i], "
                "used only for l = 1, 2, 3"
            ),
            "path_index_rule": "i is previous layer node index, j is next layer node index",
            "normalization": "energy normalization after each propagation layer",
            "coordinate_trainable": False,
            "layer_phase_removed": False,
            "output_node_phase_bias_added": True,
            "sigmoid_transmittance_removed": True,
            "raw_beta_removed": True,
            "a_trans_directly_trainable": True,
            "a_trans_init_value": A_TRANS_INIT_VALUE,
            "a_trans_init_std": A_TRANS_INIT_STD,
            "trainable_parameters": (
                "path_theta10, path_theta21, path_theta32, path_theta43, "
                "path_a_trans10, path_a_trans21, path_a_trans32, path_a_trans43, "
                "bias_phase1, bias_phase2, bias_phase3"
            ),
            "expected_trainable_parameters": expected_params,
            "best_val_acc": best_val_acc,
            "best_val_epoch": best_val_epoch,
            "final_epoch": final_epoch,
            "training_converged": training_converged,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta_loss": EARLY_STOPPING_MIN_DELTA_LOSS,
            "early_stopping_min_delta_acc": EARLY_STOPPING_MIN_DELTA_ACC,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "correct": correct,
            "total": total,
            "seed": SEED,
            "split_manifest_path": str(SPLIT_MANIFEST_PATH),
            "test_set_rule": "Use torchvision.datasets.MNIST(train=False) for inference forward",
        },
        save_path,
    )

    return save_path


def save_visualizations(model, b_theta):
    xy0, xy1, xy2, xy3, xy4 = model.get_xy()

    plot_dir = Path(
        f"xy_path_theta_QAT_attenuation_continuous_bt_{b_theta}_seed_{SEED}"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_hole_layout_visualization(
        [xy0, xy1, xy2, xy3, xy4],
        plot_dir / "fixed_line_array_layout.png",
    )

    save_path_parameter_visualization(
        model.path_theta10,
        model.path_a_trans10,
        "Layer 0 to Layer 1",
        plot_dir / "path_theta_quantized_a_trans_continuous_10.png",
        b_theta,
    )

    save_path_parameter_visualization(
        model.path_theta21,
        model.path_a_trans21,
        "Layer 1 to Layer 2",
        plot_dir / "path_theta_quantized_a_trans_continuous_21.png",
        b_theta,
    )

    save_path_parameter_visualization(
        model.path_theta32,
        model.path_a_trans32,
        "Layer 2 to Layer 3",
        plot_dir / "path_theta_quantized_a_trans_continuous_32.png",
        b_theta,
    )

    save_path_parameter_visualization(
        model.path_theta43,
        model.path_a_trans43,
        "Layer 3 to Layer 4",
        plot_dir / "path_theta_quantized_a_trans_continuous_43.png",
        b_theta,
    )

    return plot_dir


def train_one_bit_setting(
    b_theta,
    train_loader,
    val_loader,
    test_loader,
    device,
):
    print("\n" + "=" * 90)
    print(f"Start training: b_theta={b_theta}")
    print("b_theta controls both inter-layer path phase theta and output-node bias_phase")
    print("attenuation / a_trans is continuous and only clamped to [0, 1]")
    print("=" * 90)

    set_seed(SEED)

    model = OpticalPathModel(
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
        b_theta=b_theta,
        qat_quantize_phase=QAT_QUANTIZE_PHASE,
        qat_quantize_transmittance=QAT_QUANTIZE_TRANSMITTANCE,
    ).to(device)

    initial_xy = tuple(x.detach().cpu().clone() for x in model.get_xy())

    criterion = nn.KLDivLoss(reduction="batchmean")
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA,
    )

    expected_params = (
        2 * (
            N1 * N0
            + N2 * N1
            + N3 * N2
            + N4 * N3
        )
        + N1
        + N2
        + N3
    )

    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params)}")
    print(f"Expected trainable parameters: {expected_params}")
    print(f"Delta_theta = {2.0 * math.pi / (2 ** b_theta):.6e} rad")
    print("Delta_a = continuous, no attenuation quantization")
    print("\nEpoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | LR")

    best_val_acc = 0.0
    best_val_epoch = 0
    best_state = None
    best_early_stop_val_loss = float("inf")
    best_early_stop_val_acc = 0.0
    epochs_without_improvement = 0
    stopped_early = False
    final_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        final_epoch = epoch
        train_loss, train_acc = run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
        )

        val_loss, val_acc = run_one_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:5d} | "
            f"{train_loss:10.4f} | "
            f"{train_acc * 100:13.2f} | "
            f"{val_loss:8.4f} | "
            f"{val_acc * 100:10.2f} | "
            f"LR: {current_lr:.6e}"
        )

        scheduler.step()

        loss_improved = (
            val_loss < best_early_stop_val_loss - EARLY_STOPPING_MIN_DELTA_LOSS
        )
        acc_improved = (
            val_acc > best_early_stop_val_acc + EARLY_STOPPING_MIN_DELTA_ACC
        )

        if loss_improved:
            best_early_stop_val_loss = val_loss
        if acc_improved:
            best_early_stop_val_acc = val_acc

        if loss_improved or acc_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            print(
                "Early stopping: validation loss did not decrease and "
                "validation accuracy did not increase for "
                f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            print(f"Training stopped at epoch {final_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        clamp_a_trans_parameters(model)

    test_loss, test_acc, correct, total = evaluate_with_count(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\nFinal Results")
    print(f"b_theta: {b_theta}")
    print("attenuation: continuous")
    print(f"Final epoch: {final_epoch}")
    print(f"Best validation epoch: {best_val_epoch}")
    print(f"Training converged by early-stopping criterion: {stopped_early}")
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Correct predictions: {correct} / {total}")

    save_path = save_trained_model(
        model=model,
        initial_xy=initial_xy,
        b_theta=b_theta,
        expected_params=expected_params,
        best_val_acc=best_val_acc,
        best_val_epoch=best_val_epoch,
        final_epoch=final_epoch,
        training_converged=stopped_early,
        test_loss=test_loss,
        test_acc=test_acc,
        correct=correct,
        total=total,
    )

    plot_dir = save_visualizations(model, b_theta)

    print(f"Saved model to: {save_path.resolve()}")
    print(f"Visualizations saved to: {plot_dir.resolve()}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "b_theta": b_theta,
        "phase_quantizer_shared_by_path_and_bias": True,
        "attenuation_continuous": True,
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "final_epoch": final_epoch,
        "training_converged": stopped_early,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "correct": correct,
        "total": total,
        "save_path": str(save_path),
        "split_manifest_path": str(SPLIT_MANIFEST_PATH),
    }


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    b_theta = int(checkpoint["b_theta"])

    model = OpticalPathModel(
        n0=N0,
        n1=N1,
        n2=N2,
        n3=N3,
        n4=N4,
        wavelength=checkpoint.get("wavelength", WAVELENGTH),
        z0=checkpoint.get("z0", Z0),
        z1=checkpoint.get("z1", Z1),
        z2=checkpoint.get("z2", Z2),
        z3=checkpoint.get("z3", Z3),
        z4=checkpoint.get("z4", Z4),
        aperture_size=checkpoint.get("aperture_size", APERTURE_SIZE),
        use_log_intensity=USE_LOG_INTENSITY,
        b_theta=b_theta,
        qat_quantize_phase=checkpoint.get("qat_quantize_phase", QAT_QUANTIZE_PHASE),
        qat_quantize_transmittance=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    clamp_a_trans_parameters(model)
    model.eval()

    return model, checkpoint


def infer_on_test_set(checkpoint_path, device, batch_size=BATCH_SIZE):
    print(f"Loading checkpoint: {checkpoint_path}")

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    test_loader = make_test_loader_for_inference(device, batch_size=batch_size)

    criterion = nn.KLDivLoss(reduction="batchmean")

    test_loss, test_acc, correct, total = evaluate_with_count(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\nInference on fixed MNIST test set")
    print(f"Checkpoint b_theta: {checkpoint['b_theta']}")
    print("Checkpoint attenuation: continuous")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Correct predictions: {correct} / {total}")

    return {
        "checkpoint": str(checkpoint_path),
        "b_theta": checkpoint["b_theta"],
        "attenuation_continuous": True,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "correct": correct,
        "total": total,
    }


def train_all():
    global SEED
    global SPLIT_MANIFEST_PATH
    global SUMMARY_PATH

    configure_cuda()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available. Running on CPU.")

    results = []
    total_runs = len(SEEDS) * len(B_THETA_VALUES)
    run_index = 0

    for current_seed in SEEDS:
        SEED = current_seed
        SPLIT_MANIFEST_PATH = Path(
            f"main_optics_theta_QAT_attenuation_continuous_dataset_split_manifest_seed_{SEED}.pt"
        )
        SUMMARY_PATH = Path(
            f"main_optics_theta_QAT_attenuation_continuous_summary_seed_{SEED}.pt"
        )

        set_seed(SEED)
        train_loader, val_loader, test_loader = make_datasets_and_save_split_manifest(device)
        seed_results = []

        for b_theta in B_THETA_VALUES:
            run_index += 1
            print(
                f"\nRunning setting {run_index}/{total_runs}: "
                f"seed={SEED}, b_theta={b_theta}, attenuation=continuous"
            )

            result = train_one_bit_setting(
                b_theta=b_theta,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
            )
            result["seed"] = SEED
            results.append(result)
            seed_results.append(result)

        torch.save(seed_results, SUMMARY_PATH)
        print(f"Seed {SEED} summary saved to: {SUMMARY_PATH.resolve()}")

        del train_loader, val_loader, test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_rows = []
    print("\n" + "=" * 90)
    print("Three-seed accuracy statistics")
    print("b_theta | mean acc (%) | sample std (%)")

    for result in results:
        csv_rows.append(
            {
                "record_type": "seed_result",
                "seed": result["seed"],
                "b_theta": result["b_theta"],
                "final_epoch": result["final_epoch"],
                "best_val_epoch": result["best_val_epoch"],
                "training_converged": result["training_converged"],
                "best_val_acc_percent": result["best_val_acc"] * 100.0,
                "test_acc_percent": result["test_acc"] * 100.0,
                "mean_test_acc_percent": "",
                "sample_std_test_acc_percent": "",
                "correct": result["correct"],
                "total": result["total"],
                "model_path": result["save_path"],
            }
        )

    for b_theta in B_THETA_VALUES:
        bit_results = [result for result in results if result["b_theta"] == b_theta]
        accuracies_percent = np.asarray(
            [result["test_acc"] * 100.0 for result in bit_results],
            dtype=np.float64,
        )
        mean_acc_percent = float(np.mean(accuracies_percent))
        sample_std_acc_percent = float(np.std(accuracies_percent, ddof=1))

        csv_rows.append(
            {
                "record_type": "aggregate",
                "seed": "",
                "b_theta": b_theta,
                "final_epoch": "",
                "best_val_epoch": "",
                "training_converged": "",
                "best_val_acc_percent": "",
                "test_acc_percent": "",
                "mean_test_acc_percent": mean_acc_percent,
                "sample_std_test_acc_percent": sample_std_acc_percent,
                "correct": "",
                "total": "",
                "model_path": "",
            }
        )

        print(
            f"{b_theta:7d} | "
            f"{mean_acc_percent:12.6f} | "
            f"{sample_std_acc_percent:14.8f}"
        )

    fieldnames = [
        "record_type",
        "seed",
        "b_theta",
        "final_epoch",
        "best_val_epoch",
        "training_converged",
        "best_val_acc_percent",
        "test_acc_percent",
        "mean_test_acc_percent",
        "sample_std_test_acc_percent",
        "correct",
        "total",
        "model_path",
    ]
    with CSV_SAVE_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 90)
    print("All phase-QAT bit settings finished.")
    print("b_theta was shared by inter-layer path phase and output-node bias phase.")
    print("attenuation / a_trans stayed continuous.")
    print(f"CSV saved to: {CSV_SAVE_PATH.resolve()}")
    print("=" * 90)

    print("\nSummary")
    print("seed | b_theta | final epoch | best val epoch | converged | test acc (%)")

    for result in results:
        print(
            f"{result['seed']:4d} | "
            f"{result['b_theta']:7d} | "
            f"{result['final_epoch']:11d} | "
            f"{result['best_val_epoch']:14d} | "
            f"{str(result['training_converged']):9s} | "
            f"{result['test_acc'] * 100:12.6f}"
        )


def main():
    configure_cuda()

    if MODE == "train":
        train_all()
        return

    if MODE == "infer":
        if INFER_CHECKPOINT_PATH is None:
            raise ValueError("Set INFER_CHECKPOINT_PATH before running MODE='infer'.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        infer_on_test_set(
            checkpoint_path=Path(INFER_CHECKPOINT_PATH),
            device=device,
            batch_size=BATCH_SIZE,
        )
        return

    raise ValueError("MODE must be 'train' or 'infer'.")


if __name__ == "__main__":
    main()

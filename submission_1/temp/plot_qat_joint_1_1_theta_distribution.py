from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


# This script is a reduced version of Phase_theta_zhifang_fig.py.
# It calculates and saves only the QAT-joint, (1,1) phase subfigure.
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = (
    SCRIPT_DIR.parent
    / "New_version_1.0"
    / "main_optics_bit_QAT_joint_shared_theta_bias_phase_1.1_bt_1_ba_1.pt"
)
OUTPUT_PATH = SCRIPT_DIR / "Figs" / "qat_joint_theta_distribution.pdf"

THETA_PERIOD = 2.0 * np.pi
NUM_BINS = 80
B_THETA = 1


def load_checkpoint(pt_path):
    if not pt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pt_path.resolve()}")

    return torch.load(pt_path, map_location="cpu")


def extract_state_dict(checkpoint):
    """Extract a state dictionary from the checkpoint formats used by the source code."""
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "net_state_dict", "model"]:
            if key in checkpoint:
                value = checkpoint[key]

                if isinstance(value, dict):
                    return value

                if hasattr(value, "state_dict"):
                    return value.state_dict()

        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint

    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()

    raise TypeError("Unsupported checkpoint format. Please inspect the .pt file structure.")


def find_theta_tensor_from_state_dict(state_dict):
    """Find the phase tensor using the same matching rules as the source code."""
    theta_key_candidates = ["theta_ij", "theta", "phase", "phi"]
    matched_items = []

    for key, value in state_dict.items():
        if torch.is_tensor(value) and any(
            candidate in key.lower() for candidate in theta_key_candidates
        ):
            matched_items.append((key, value.detach().cpu()))

    if not matched_items:
        raise KeyError(
            "No theta-related tensor found. Update theta_key_candidates or specify "
            "the exact theta key."
        )

    if len(matched_items) == 1:
        return matched_items[0][1]

    for key, value in matched_items:
        if "theta_ij" in key.lower():
            return value

    return torch.cat([value.reshape(-1) for _, value in matched_items], dim=0)


def load_theta_from_pt(pt_path):
    checkpoint = load_checkpoint(pt_path)
    state_dict = extract_state_dict(checkpoint)
    theta_tensor = find_theta_tensor_from_state_dict(state_dict)
    theta = theta_tensor.detach().cpu().numpy().reshape(-1)
    theta = theta[np.isfinite(theta)]

    if theta.size == 0:
        raise ValueError(f"Empty theta tensor loaded from {pt_path}")

    return theta


def wrap_theta_to_0_2pi(theta):
    return np.mod(theta, THETA_PERIOD)


def quantize_theta_uniform(theta, num_bits):
    theta_wrapped = wrap_theta_to_0_2pi(theta)
    num_levels = 2 ** int(num_bits)
    step = THETA_PERIOD / num_levels
    indices = np.round(theta_wrapped / step).astype(np.int64)
    indices = np.mod(indices, num_levels)
    return indices * step


def save_qat_joint_1_1_theta_figure(theta, save_path):
    theta_pi = wrap_theta_to_0_2pi(theta) / np.pi
    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    ax.hist(
        theta_pi,
        bins=NUM_BINS,
        range=(0.0, 2.0),
        density=True,
        alpha=0.78,
        color="blue",
        edgecolor="black",
        linewidth=0.25,
    )
    ax.set_title("QAT-joint, (1,1), quant", fontsize=9)
    ax.set_xlim(0.0, 2.0)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["0", r"$0.5\pi$", r"$\pi$", r"$1.5\pi$", r"$2\pi$"])
    ax.set_xlabel(r"$\theta_{ij}$")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    qat_theta = load_theta_from_pt(CHECKPOINT_PATH)
    qat_theta_quantized = quantize_theta_uniform(qat_theta, B_THETA)
    save_qat_joint_1_1_theta_figure(qat_theta_quantized, OUTPUT_PATH)


if __name__ == "__main__":
    main()

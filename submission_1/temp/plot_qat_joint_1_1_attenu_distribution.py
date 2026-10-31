from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


# This script is a reduced version of Attenu_alpha_zhifang_fig.py.
# It calculates and saves only the QAT-joint, (1,1) attenuation subfigure.
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = (
    SCRIPT_DIR.parent
    / "New_version_1.0"
    / "main_optics_bit_QAT_joint_shared_theta_bias_phase_1.1_bt_1_ba_1.pt"
)
OUTPUT_PATH = SCRIPT_DIR / "Figs" / "qat_joint_attenu_distribution.pdf"

ATTENU_MIN = 0.0
ATTENU_MAX = 1.0
NUM_BINS = 80
B_ALPHA = 1


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


def find_attenu_tensor_from_state_dict(state_dict):
    """Find attenuation tensors using the same matching rules as the source code."""
    attenu_key_candidates = [
        "path_a_trans10_continuous",
        "path_a_trans21_continuous",
        "path_a_trans32_continuous",
        "path_a_trans43_continuous",
        "path_a_trans10",
        "path_a_trans21",
        "path_a_trans32",
        "path_a_trans43",
        "path_a_trans",
        "a_trans",
        "raw_beta",
        "beta",
    ]
    matched_items = []

    for key, value in state_dict.items():
        if torch.is_tensor(value) and any(
            candidate in key.lower() for candidate in attenu_key_candidates
        ):
            matched_items.append((key, value.detach().cpu()))

    if not matched_items:
        raise KeyError(
            "No attenuation-related tensor found. Update attenu_key_candidates or "
            "specify the exact attenuation key."
        )

    if len(matched_items) == 1:
        return matched_items[0][1]

    for key, value in matched_items:
        key_lower = key.lower()
        if "alpha_ij" in key_lower or "a_ij" in key_lower:
            return value

    return torch.cat([value.reshape(-1) for _, value in matched_items], dim=0)


def load_attenu_from_pt(pt_path):
    checkpoint = load_checkpoint(pt_path)
    state_dict = extract_state_dict(checkpoint)
    attenu_tensor = find_attenu_tensor_from_state_dict(state_dict)
    attenu = attenu_tensor.detach().cpu().numpy().reshape(-1)
    attenu = attenu[np.isfinite(attenu)]

    if attenu.size == 0:
        raise ValueError(f"Empty attenuation tensor loaded from {pt_path}")

    return attenu


def clip_attenu_to_0_1(attenu):
    return np.clip(attenu, ATTENU_MIN, ATTENU_MAX)


def quantize_attenu_uniform(attenu, num_bits):
    attenu_clipped = clip_attenu_to_0_1(attenu)
    num_levels = 2 ** int(num_bits)

    if num_levels <= 1:
        return np.zeros_like(attenu_clipped)

    step = (ATTENU_MAX - ATTENU_MIN) / (num_levels - 1)
    indices = np.round((attenu_clipped - ATTENU_MIN) / step).astype(np.int64)
    indices = np.clip(indices, 0, num_levels - 1)
    return ATTENU_MIN + indices * step


def save_qat_joint_1_1_attenu_figure(attenu, save_path):
    attenu_clipped = clip_attenu_to_0_1(attenu)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    ax.hist(
        attenu_clipped,
        bins=NUM_BINS,
        range=(ATTENU_MIN, ATTENU_MAX),
        density=True,
        alpha=0.78,
        color="blue",
        edgecolor="black",
        linewidth=0.25,
    )
    ax.set_title("QAT-joint, (1,1), quant", fontsize=9)
    ax.set_xlim(ATTENU_MIN, ATTENU_MAX)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel(r"$a_{ij}$")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    qat_attenu = load_attenu_from_pt(CHECKPOINT_PATH)
    qat_attenu_quantized = quantize_attenu_uniform(qat_attenu, B_ALPHA)
    save_qat_joint_1_1_attenu_figure(qat_attenu_quantized, OUTPUT_PATH)


if __name__ == "__main__":
    main()

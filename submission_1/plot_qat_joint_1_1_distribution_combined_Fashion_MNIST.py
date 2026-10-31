from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = (
    SCRIPT_DIR
    / "figs"
    / "fashion_mnist_qat_joint_attenu_theta_distribution_3_seeds_combined_2.pdf"
)
DATASET_NAME = "Fashion-MNIST"

# Values reproduced from the existing three-seed Fashion-MNIST figure.
SEEDS = [42, 123, 2024]
INFERENCE_ACCURACY = [82.22, 82.65, 82.67]
PROBABILITIES = np.array(
    [
        [0.4224, 0.5776, 0.9700, 0.0300],
        [0.4163, 0.5837, 0.9746, 0.0254],
        [0.4394, 0.5606, 0.9645, 0.0355],
    ]
)

# Okabe-Ito palette plus hatch redundancy for colour-vision deficiency.
COLORS = ["#D55E00", "#0072B2", "#009E73"]
HATCHES = ["///", "\\\\", "xx"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def create_combined_figure(save_path):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    category_x = np.array([0.0, 1.0, 2.55, 3.55])
    category_labels = [
        "$a_{ij}=0$",
        "$a_{ij}=1$",
        r"$\theta_{ij}=0$",
        r"$\theta_{ij}=\pi$",
    ]
    bar_width = 0.22
    offsets = np.array([-bar_width, 0.0, bar_width])

    for seed_index, (seed, accuracy, color, hatch, offset) in enumerate(
        zip(SEEDS, INFERENCE_ACCURACY, COLORS, HATCHES, offsets)
    ):
        bars = ax.bar(
            category_x + offset,
            PROBABILITIES[seed_index],
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
            label=f"Seed {seed} (infer. acc.: {accuracy:.2f}%)",
            zorder=3,
        )

        for bar, probability in zip(bars, PROBABILITIES[seed_index]):
            label_y = probability + 0.018
            vertical_alignment = "bottom"
            label_box = None
            if probability > 0.90:
                label_y = probability - 0.035
                vertical_alignment = "top"
                label_box = {
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                    "pad": 0.5,
                }
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                f"{probability:.2%}",
                ha="center",
                va=vertical_alignment,
                rotation=90,
                fontsize=9,
                color="#111111",
                clip_on=False,
                bbox=label_box,
            )

    ax.axvline(1.775, color="#8C8C8C", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(0.5, -0.145, "Amplitude transmittance", ha="center", va="top", fontsize=9)
    ax.text(3.05, -0.145, "Phase", ha="center", va="top", fontsize=9)

    ax.set_title(
        rf"{DATASET_NAME} QAT-joint ($b_\theta=1$, $b_a=1$)",
        pad=8,
    )
    ax.set_ylabel("Probability")
    ax.set_xticks(category_x)
    ax.set_xticklabels(category_labels)
    ax.set_xlim(-0.55, 4.10)
    ax.set_ylim(0.0, 1.12)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(axis="y", color="#C7C7C7", linewidth=0.6, alpha=0.65, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        edgecolor="#444444",
        framealpha=1.0,
    )

    fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.25)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    create_combined_figure(OUTPUT_PATH)
    print(f"Saved combined figure: {OUTPUT_PATH.resolve()}")

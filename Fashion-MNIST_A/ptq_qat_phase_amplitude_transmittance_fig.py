from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Output settings
# ============================================================
OUTPUT_DIR = Path("./figs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUTPUT_DIR / "fashion_mnist_phase_amplitude_transmittance_accuracy.pdf"
PNG_PATH = OUTPUT_DIR / "fashion_mnist_phase_amplitude_transmittance_accuracy.png"


# ============================================================
# Raw test accuracies for seeds {42, 123, 2024}
# Unit: %
# ============================================================
bit_widths = np.array([1, 2, 3, 6])
x_positions = np.arange(len(bit_widths))

full_precision_accuracy = np.array(
    [83.94, 84.11, 83.72],
    dtype=np.float64,
)
full_precision_mean = np.mean(full_precision_accuracy)

# Phase-only quantization
ptq_phase_accuracy = np.array(
    [
        [57.88, 68.57, 62.85],  # b_theta = 1
        [81.90, 81.71, 76.94],  # b_theta = 2
        [84.16, 83.52, 83.06],  # b_theta = 3
        [83.93, 84.16, 83.70],  # b_theta = 6
    ],
    dtype=np.float64,
)

qat_phase_accuracy = np.array(
    [
        [83.02, 82.81, 83.05],  # b_theta = 1
        [83.76, 83.19, 83.27],  # b_theta = 2
        [83.74, 83.88, 83.69],  # b_theta = 3
        [84.02, 84.08, 83.59],  # b_theta = 6
    ],
    dtype=np.float64,
)

# Amplitude-transmittance-only quantization
ptq_attenuation_accuracy = np.array(
    [
        [83.17, 83.44, 83.05],  # b_a = 1
        [84.01, 84.04, 83.50],  # b_a = 2
        [84.03, 84.26, 83.73],  # b_a = 3
        [83.97, 84.14, 83.73],  # b_a = 6
    ],
    dtype=np.float64,
)

qat_attenuation_accuracy = np.array(
    [
        [83.87, 83.72, 83.87],  # b_a = 1
        [83.91, 84.21, 83.75],  # b_a = 2
        [84.05, 83.99, 83.82],  # b_a = 3
        [83.92, 83.93, 84.14],  # b_a = 6
    ],
    dtype=np.float64,
)


# ============================================================
# Statistical calculation
# ============================================================
def calculate_statistics(accuracy):
    """
    Calculate the mean and sample standard deviation across
    the three random seeds.

    Parameters
    ----------
    accuracy : np.ndarray
        Shape: (number of bit widths, number of random seeds).

    Returns
    -------
    mean_accuracy : np.ndarray
        Mean accuracy for each bit width.
    sample_std : np.ndarray
        Sample standard deviation for each bit width (ddof=1).
    """
    mean_accuracy = np.mean(accuracy, axis=1)
    sample_std = np.std(accuracy, axis=1, ddof=1)

    return mean_accuracy, sample_std


ptq_phase_mean, ptq_phase_std = calculate_statistics(
    ptq_phase_accuracy
)
qat_phase_mean, qat_phase_std = calculate_statistics(
    qat_phase_accuracy
)

ptq_attenuation_mean, ptq_attenuation_std = calculate_statistics(
    ptq_attenuation_accuracy
)
qat_attenuation_mean, qat_attenuation_std = calculate_statistics(
    qat_attenuation_accuracy
)


# ============================================================
# Print statistics
# ============================================================
def print_statistics(method, bits, means, stds):
    print("\n" + method)
    print("bit | mean accuracy (%) | sample std (%)")

    for bit, mean, std in zip(bits, means, stds):
        print(f"{bit:3d} | {mean:17.4f} | {std:14.4f}")


print_statistics(
    "PTQ-phase",
    bit_widths,
    ptq_phase_mean,
    ptq_phase_std,
)

print_statistics(
    "QAT-phase",
    bit_widths,
    qat_phase_mean,
    qat_phase_std,
)

print_statistics(
    "PTQ-amplitude",
    bit_widths,
    ptq_attenuation_mean,
    ptq_attenuation_std,
)

print_statistics(
    "QAT-amplitude",
    bit_widths,
    qat_attenuation_mean,
    qat_attenuation_std,
)


# ============================================================
# Plot settings
# ============================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(7.2, 3.2),
    constrained_layout=True,
)

phase_ax = axes[0]
attenuation_ax = axes[1]


# ============================================================
# (a) Phase-only quantization
# ============================================================
phase_ax.errorbar(
    x_positions,
    ptq_phase_mean,
    yerr=ptq_phase_std,
    color="#1f77b4",
    marker="o",
    markersize=5,
    markerfacecolor="white",
    markeredgewidth=1.2,
    linewidth=1.5,
    capsize=3.5,
    capthick=1.0,
    elinewidth=1.0,
    label="PTQ-phase",
)

phase_ax.errorbar(
    x_positions,
    qat_phase_mean,
    yerr=qat_phase_std,
    color="#d62728",
    marker="s",
    markersize=5,
    markerfacecolor="white",
    markeredgewidth=1.2,
    linewidth=1.5,
    capsize=3.5,
    capthick=1.0,
    elinewidth=1.0,
    label="QAT-phase",
)

phase_ax.axhline(
    y=full_precision_mean,
    color="black",
    linestyle="--",
    linewidth=1.3,
    label="Full precision",
)

phase_ax.set_xlabel(r"Phase bit width $b_{\theta}$")
phase_ax.set_ylabel("Test accuracy (%)")
phase_ax.set_xticks(x_positions)
phase_ax.set_xticklabels(bit_widths)
phase_ax.set_xlim(-0.25, len(bit_widths) - 0.75)
phase_ax.set_title("(a) Phase-only quantization")
phase_ax.grid(
    True,
    which="major",
    linestyle="--",
    linewidth=0.6,
    alpha=0.45,
)
phase_ax.legend(
    loc="lower right",
    frameon=True,
    framealpha=0.95,
)


# ============================================================
# (b) Amplitude-transmittance-only quantization
# ============================================================
attenuation_ax.errorbar(
    x_positions,
    ptq_attenuation_mean,
    yerr=ptq_attenuation_std,
    color="#1f77b4",
    marker="o",
    markersize=5,
    markerfacecolor="white",
    markeredgewidth=1.2,
    linewidth=1.5,
    capsize=3.5,
    capthick=1.0,
    elinewidth=1.0,
    label="PTQ-amplitude",
)

attenuation_ax.errorbar(
    x_positions,
    qat_attenuation_mean,
    yerr=qat_attenuation_std,
    color="#d62728",
    marker="s",
    markersize=5,
    markerfacecolor="white",
    markeredgewidth=1.2,
    linewidth=1.5,
    capsize=3.5,
    capthick=1.0,
    elinewidth=1.0,
    label="QAT-amplitude",
)

attenuation_ax.axhline(
    y=full_precision_mean,
    color="black",
    linestyle="--",
    linewidth=1.3,
    label="Full precision",
)

attenuation_ax.set_xlabel(r"Amplitude-transmittance bit width $b_a$")
attenuation_ax.set_ylabel("Test accuracy (%)")
attenuation_ax.set_xticks(x_positions)
attenuation_ax.set_xticklabels(bit_widths)
attenuation_ax.set_xlim(-0.25, len(bit_widths) - 0.75)
attenuation_ax.set_title("(b) Amplitude-transmittance-only quantization")
attenuation_ax.grid(
    True,
    which="major",
    linestyle="--",
    linewidth=0.6,
    alpha=0.45,
)
attenuation_ax.legend(
    loc="lower right",
    frameon=True,
    framealpha=0.95,
)


# ============================================================
# Save and display
# ============================================================
fig.savefig(
    PDF_PATH,
    bbox_inches="tight",
)

fig.savefig(
    PNG_PATH,
    dpi=600,
    bbox_inches="tight",
)

print(f"\nPDF saved to: {PDF_PATH.resolve()}")
print(f"PNG saved to: {PNG_PATH.resolve()}")

plt.show()

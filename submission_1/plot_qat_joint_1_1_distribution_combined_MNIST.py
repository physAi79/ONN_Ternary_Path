from pathlib import Path

import numpy as np

import plot_qat_joint_1_1_distribution_combined_Fashion_MNIST as combined_plot


SCRIPT_DIR = Path(__file__).resolve().parent

combined_plot.DATASET_NAME = "MNIST"
combined_plot.OUTPUT_PATH = (
    SCRIPT_DIR
    / "figs"
    / "mnist_qat_joint_attenu_theta_distribution_3_seeds_1.pdf"
)
combined_plot.INFERENCE_ACCURACY = [89.76, 89.77, 89.78]
combined_plot.PROBABILITIES = np.array(
    [
        [0.2830, 0.7170, 0.9812, 0.0188],
        [0.2881, 0.7119, 0.9788, 0.0212],
        [0.2557, 0.7443, 0.9928, 0.0072],
    ]
)


if __name__ == "__main__":
    combined_plot.create_combined_figure(combined_plot.OUTPUT_PATH)
    print(f"Saved combined MNIST figure: {combined_plot.OUTPUT_PATH.resolve()}")

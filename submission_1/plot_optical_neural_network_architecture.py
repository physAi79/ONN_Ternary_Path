from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "Figs" / "optical_neural_network_architecture.pdf"

# Nature-style, double-column final artwork: approximately 183 mm wide.
FINAL_FIGURE_SIZE_IN = (7.20, 3.80)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


LAYER_SPECS = [
    ("", 784, r"$\psi^{(0)}$", "#56B4E9"),
    ("", 64, r"$\psi^{(1)}$", "#009E73"),
    ("", 128, r"$\psi^{(2)}$", "#E69F00"),
    ("", 64, r"$\psi^{(3)}$", "#CC79A7"),
    ("", 10, r"$\psi^{(4)}$", "#D55E00"),
]

LAYER_X = [2.7, 5.1, 7.5, 9.9, 12.3]

# Conversion for point-sized layout adjustments in the equal-aspect diagram.
# One diagram coordinate unit is approximately 3 cm.
DATA_UNITS_PER_POINT = 2.54 / (72.0 * 3.0)

# The canvas is authored at final publication width, so point sizes below are
# also the effective sizes in the submitted figure.
FONT_SCALE = 1.0


def representative_node_positions(node_count, center_y=3.65):
    """Return representative node positions for a compact conventional NN diagram."""
    if node_count <= 10:
        spacing = 0.58
        start = center_y + spacing * (node_count - 1) / 2.0
        return [start - spacing * index for index in range(node_count)], None

    positions = [6.25, 5.65, 5.05, 2.25, 1.65, 1.05]
    return positions, center_y


def draw_dense_connections(ax, x_left, y_left, x_right, y_right):
    for source_y in y_left:
        for target_y in y_right:
            ax.plot(
                [x_left + 0.13, x_right - 0.13],
                [source_y, target_y],
                color="#5F6368",
                linewidth=0.70,
                alpha=0.50,
                zorder=1,
            )


def draw_layer(ax, x, title, node_count, state_label, facecolor, show_ellipsis=True):
    node_y, ellipsis_y = representative_node_positions(node_count)

    for y in node_y:
        ax.add_patch(
            Circle(
                (x, y),
                radius=0.13,
                facecolor=facecolor,
                edgecolor="#263238",
                linewidth=0.8,
                zorder=4,
            )
        )

    if ellipsis_y is not None and show_ellipsis:
        ax.text(x, ellipsis_y, r"$\vdots$", ha="center", va="center", fontsize=20 * FONT_SCALE)

    if title:
        ax.text(
            x,
            7.52,
            title,
            ha="center",
            va="bottom",
            fontsize=12.5 * FONT_SCALE,
            fontweight="bold",
        )
    ax.text(
        x,
        0.45 + 10.0 * DATA_UNITS_PER_POINT,
        state_label,
        ha="center",
        va="top",
        fontsize=12 * FONT_SCALE,
    )

    return node_y


def draw_input_encoding(ax):
    input_box_shift_x = -60.0 * DATA_UNITS_PER_POINT
    input_box_left_extension = 40.0 * DATA_UNITS_PER_POINT
    input_box_symmetric_extension = 10.0 * DATA_UNITS_PER_POINT
    input_box_vertical_extension = 40.0 * DATA_UNITS_PER_POINT
    box = FancyBboxPatch(
        (
            0.25
            + input_box_shift_x
            - input_box_left_extension
            - input_box_symmetric_extension,
            2.65 - input_box_vertical_extension,
        ),
        1.45 + input_box_left_extension + 2.0 * input_box_symmetric_extension,
        2.0 + 2.0 * input_box_vertical_extension,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="#E6E6E6",
        edgecolor="#000000",
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(box)

    # Stylised 28 x 28 grayscale image.
    grid_origin_x, grid_origin_y = 0.46 + input_box_shift_x, 3.35
    cell = 0.125
    grey_values = [
        [0.95, 0.85, 0.42, 0.18, 0.72, 0.95],
        [0.92, 0.34, 0.12, 0.38, 0.90, 0.96],
        [0.86, 0.22, 0.62, 0.16, 0.68, 0.94],
        [0.93, 0.72, 0.82, 0.20, 0.36, 0.91],
        [0.96, 0.88, 0.44, 0.16, 0.70, 0.95],
        [0.98, 0.94, 0.76, 0.50, 0.88, 0.98],
    ]
    for row_index, row in enumerate(grey_values):
        for column_index, value in enumerate(row):
            ax.add_patch(
                Rectangle(
                    (
                        grid_origin_x + column_index * cell,
                        grid_origin_y + (5 - row_index) * cell,
                    ),
                    cell,
                    cell,
                    facecolor=str(value),
                    edgecolor="none",
                    zorder=4,
                )
            )

    ax.text(
        0.975 + input_box_shift_x - 12.0 * DATA_UNITS_PER_POINT,
        4.38 + 83.0 * DATA_UNITS_PER_POINT,
        "28 x 28 image",
        ha="center",
        va="center",
        fontsize=10.5 * FONT_SCALE,
    )
    ax.text(
        0.975 + input_box_shift_x - 20.0 * DATA_UNITS_PER_POINT,
        3.10 - 15.0 * DATA_UNITS_PER_POINT,
        "Flatten",
        ha="center",
        va="center",
        fontsize=10 * FONT_SCALE,
    )
    ax.text(
        0.975 + input_box_shift_x - 20.0 * DATA_UNITS_PER_POINT,
        2.84 - 27.0 * DATA_UNITS_PER_POINT,
        r"$x_i\in[0,1]$",
        ha="center",
        va="center",
        fontsize=8.5 * FONT_SCALE,
    )

    arrow = FancyArrowPatch(
        (1.72 - 40.0 * DATA_UNITS_PER_POINT, 3.65),
        (2.48 + 15.0 * DATA_UNITS_PER_POINT, 3.65),
        arrowstyle="-|>",
        mutation_scale=12,
        color="#000000",
        linewidth=1.1,
    )
    ax.add_patch(arrow)
    ax.text(
        2.08 + 10.0 * DATA_UNITS_PER_POINT,
        4.16,
        "Phase encoding",
        ha="center",
        va="bottom",
        fontsize=5.5 * FONT_SCALE,
    )
    ax.text(
        2.08 - 17.0 * DATA_UNITS_PER_POINT,
        3.78 - 10.0 * DATA_UNITS_PER_POINT,
        r"$\widetilde{\psi}^{(0)}_i=e^{\mathrm{i}\pi x_i}$",
        ha="center",
        va="bottom",
        fontsize=6 * FONT_SCALE,
    )


def draw_output_readout(ax):
    # The equal-aspect diagram uses approximately 3 cm per coordinate unit.
    output_box_shift_x = 56.0 * DATA_UNITS_PER_POINT
    output_box_symmetric_extension = 29.0 * DATA_UNITS_PER_POINT
    formula_extra_shift_x = 70.0 * DATA_UNITS_PER_POINT
    detector_title_x = 14.42 + 30.0 * DATA_UNITS_PER_POINT
    formula_x = detector_title_x + formula_extra_shift_x
    box = FancyBboxPatch(
        (
            13.55
            + 30.0 * DATA_UNITS_PER_POINT
            + output_box_shift_x
            - output_box_symmetric_extension,
            2.47 - 15.0 * DATA_UNITS_PER_POINT,
        ),
        1.40
        + 50.0 * DATA_UNITS_PER_POINT
        + 2.0 * output_box_symmetric_extension,
        2.36,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        facecolor="#F6E58D",
        edgecolor="#000000",
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(box)

    arrow = FancyArrowPatch(
        (12.50, 3.65),
        (13.48 + 30.0 * DATA_UNITS_PER_POINT, 3.65),
        arrowstyle="-|>",
        mutation_scale=12,
        color="#000000",
        linewidth=1.1,
    )
    ax.add_patch(arrow)
    normalisation_x = 13.02 + 25.0 * DATA_UNITS_PER_POINT
    ax.text(normalisation_x, 4.18, "Global energy", ha="center", va="bottom", fontsize=5 * FONT_SCALE)
    ax.text(normalisation_x, 3.93, "normalisation", ha="center", va="bottom", fontsize=5 * FONT_SCALE)

    ax.text(detector_title_x, 5.03, "Square-law detection", ha="center", fontsize=8 * FONT_SCALE, fontweight="bold")
    ax.text(formula_x, 4.12, r"$I_j=|\psi_j^{(4)}|^2$", ha="center", fontsize=7.5 * FONT_SCALE)
    ax.text(formula_x, 3.66, r"$s_j=\log(I_j+\varepsilon)$", ha="center", fontsize=7.5 * FONT_SCALE)
    ax.text(formula_x, 3.20, r"$p_j=\mathrm{softmax}(\mathbf{s})_j$", ha="center", fontsize=7.5 * FONT_SCALE)
    ax.text(formula_x, 2.78, r"$\widehat{y}=\arg\max_j I_j$", ha="center", fontsize=7.5 * FONT_SCALE)


def create_architecture_figure(save_path):
    fig, ax = plt.subplots(figsize=FINAL_FIGURE_SIZE_IN)
    fig.patch.set_facecolor("white")

    layer_nodes = []
    for layer_position, (x, (title, count, state_label, color)) in enumerate(
        zip(LAYER_X, LAYER_SPECS)
    ):
        layer_nodes.append(
            draw_layer(
                ax,
                x,
                title,
                count,
                state_label,
                color,
                show_ellipsis=layer_position not in (1, 2, 3),
            )
        )

    for layer_index in range(len(LAYER_X) - 1):
        draw_dense_connections(
            ax,
            LAYER_X[layer_index],
            layer_nodes[layer_index],
            LAYER_X[layer_index + 1],
            layer_nodes[layer_index + 1],
        )

        midpoint = (LAYER_X[layer_index] + LAYER_X[layer_index + 1]) / 2.0
        ax.text(
            midpoint,
            6.72,
            rf"$\mathbf{{K}}_q^{{({layer_index + 1})}}$",
            ha="center",
            va="center",
            fontsize=13 * FONT_SCALE,
            color="#000000",
        )

    # Hidden-layer phase biases are multiplicative rotations, not activations.
    for x, layer_index in zip(LAYER_X[1:4], [1, 2, 3]):
        ax.text(
            x,
            -0.11 - 22.0 * DATA_UNITS_PER_POINT,
            rf"$\mathbf{{B}}^{{({layer_index})}}$" + "\nphase bias",
            ha="center",
            va="center",
            fontsize=8.5 * FONT_SCALE,
            color="#000000",
            linespacing=0.9,
        )

    draw_input_encoding(ax)
    draw_output_readout(ax)

    ax.text(
        7.55,
        8.72 - 30.0 * DATA_UNITS_PER_POINT,
        "Coherent complex linear optical neural network",
        ha="center",
        va="center",
        fontsize=14.5 * FONT_SCALE,
        fontweight="bold",
    )
    # Include the full input box after its leftward shift and 40 pt left-edge
    # extension; the previous lower limit of 0 clipped the box and its content.
    ax.set_xlim(-1.10, 17.35)
    ax.set_ylim(-0.82, 9.02)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print(f"Generated optical neural network architecture: {save_path.resolve()}")


def main():
    create_architecture_figure(OUTPUT_PATH)


if __name__ == "__main__":
    main()

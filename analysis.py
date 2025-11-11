"""
analysis.py

Utility script for comparing an original (unprotected) image, a glazed image,
and an AI-generated result. Computes summary statistics and histogram plots
to help inspect the glazing signal.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend for non-GUI environments
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
try:
    import pandas as pd  # optional; fallback to CSV writer if unavailable
except Exception:
    pd = None
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    """Load an image as an RGB numpy array in range [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def compute_stats(arr: np.ndarray) -> dict:
    """Compute mean, std, and pixel-level noise metric."""
    mean_color = np.mean(arr, axis=(0, 1))
    std_color = np.std(arr, axis=(0, 1))
    noise_level = np.mean(np.abs(arr - np.round(arr * 255.0) / 255.0))
    return {
        "Mean R": mean_color[0],
        "Mean G": mean_color[1],
        "Mean B": mean_color[2],
        "Std R": std_color[0],
        "Std G": std_color[1],
        "Std B": std_color[2],
        "Pixel Noise": noise_level,
    }


def plot_histograms(arrays: list[np.ndarray], labels: list[str], output: Path | None) -> None:
    """Plot RGB histograms for each image."""
    channels = ["Red", "Green", "Blue"]
    fig, axes = plt.subplots(len(arrays), 3, figsize=(12, 4 * len(arrays)))

    if len(arrays) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, arr in enumerate(arrays):
        for j in range(3):
            axes[i, j].hist(
                arr[:, :, j].flatten(),
                bins=50,
                color=channels[j].lower(),
                alpha=0.8,
            )
            axes[i, j].set_title(f"{channels[j]} - {labels[i]}")
            axes[i, j].set_xlim(0, 1)
            axes[i, j].set_xlabel("Normalized intensity")
            axes[i, j].set_ylabel("Frequency")

    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=200)
    else:
        plt.show()

    plt.close(fig)


def save_stats_image(stats: list[dict], labels: list[str], columns: list[str], output: Path) -> None:
    """Render statistics as a table image."""
    fig_width = max(6.0, len(columns) * 1.2)
    fig_height = max(2.0, 1.0 + 0.6 * len(stats))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    cell_text = []
    for row in stats:
        cell_text.append([f"{row[col]:0.6f}" for col in columns])

    table = ax.table(
        cellText=cell_text,
        rowLabels=labels,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.3)

    ax.set_title("Glaze Protection Analysis", pad=20, fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_heatmap_overlay(image_path: Path, output: Path) -> None:
    """Create a heatmap overlay visualization of perturbations."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)

    blurred = gaussian_filter(arr, sigma=(3, 3, 0))
    residual = np.abs(arr - blurred)
    heatmap = np.mean(residual, axis=2)

    # Normalize for visualization
    max_val = float(heatmap.max())
    min_val = float(heatmap.min())
    if max_val - min_val < 1e-12:
        heatmap_norm = np.zeros_like(heatmap)
    else:
        heatmap_norm = (heatmap - min_val) / (max_val - min_val)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    im = ax.imshow(heatmap_norm, cmap="inferno", alpha=0.5)
    ax.set_title("Glaze Perturbation Heatmap (High = Stronger Protection)")
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Perturbation Intensity")

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze glazing effects by comparing original, glazed, and AI result images."
    )
    parser.add_argument("--original", type=Path, required=True, help="Path to the original image.")
    parser.add_argument("--glazed", type=Path, required=True, help="Path to the glazed image.")
    parser.add_argument("--ai-result", type=Path, required=True, help="Path to the AI-generated image.")
    parser.add_argument(
        "--hist-output",
        type=Path,
        default=None,
        help="Optional path to save histogram plot (PNG). If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional path to save computed statistics (CSV).",
    )
    parser.add_argument(
        "--stats-image",
        type=Path,
        default=None,
        help="Optional path to save computed statistics as an image (PNG).",
    )
    parser.add_argument(
        "--heatmap-input",
        type=Path,
        default=None,
        help="Optional path for generating perturbation heatmap (defaults to glazed image).",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=None,
        help="Optional path to save perturbation heatmap overlay (PNG).",
    )

    args = parser.parse_args()

    paths = [args.original, args.glazed, args.ai_result]
    labels = ["Original", "Glazed", "AI Result"]

    arrays = [load_image(path) for path in paths]
    stats = [compute_stats(arr) for arr in arrays]
    columns = list(stats[0].keys())

    if pd is not None:
        df = pd.DataFrame(stats, index=labels)
        print("\nGlaze Protection Analysis\n")
        print(df.to_string(float_format=lambda x: f"{x:0.6f}"))
        if args.stats_output:
            df.to_csv(args.stats_output)
            print(f"\nSaved statistics to {args.stats_output.resolve()}")
        if args.stats_image:
            save_stats_image(stats, labels, columns, args.stats_image)
            print(f"Saved statistics image to {args.stats_image.resolve()}")
    else:
        # Fallback: print plain text and write basic CSV without pandas
        print("\nGlaze Protection Analysis (pandas not available)\n")
        for label, row in zip(labels, stats):
            print(f"{label}: " + ", ".join(f"{k}={v:.6f}" for k, v in row.items()))
        if args.stats_output:
            import csv
            with open(args.stats_output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(stats)
            print(f"\nSaved statistics to {args.stats_output.resolve()}")
        if args.stats_image:
            save_stats_image(stats, labels, columns, args.stats_image)
            print(f"Saved statistics image to {args.stats_image.resolve()}")

    plot_histograms(arrays, labels, args.hist_output)
    if args.hist_output:
        print(f"Saved histogram plot to {args.hist_output.resolve()}")

    if args.heatmap_output:
        heatmap_source = args.heatmap_input or args.glazed
        generate_heatmap_overlay(heatmap_source, args.heatmap_output)
        print(f"Saved heatmap overlay to {args.heatmap_output.resolve()}")


if __name__ == "__main__":
    main()


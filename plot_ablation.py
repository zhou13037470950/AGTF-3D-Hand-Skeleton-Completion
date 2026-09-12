from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


DEFAULT_SUMMARY = "runs/ablation1/results/ablation_summary.csv"
DEFAULT_OUTPUT_DIR = "runs/ablation1/results/plots"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_metric_plot(
    rows: list[dict[str, str]],
    metric: str,
    std_metric: str,
    title: str,
    xlabel: str,
    output: Path,
    smaller_is_better: bool,
) -> None:
    labels = [row.get("label") or row["variant"] for row in rows]
    values = [float(row[metric]) for row in rows]
    errors = [float(row.get(std_metric, 0.0) or 0.0) for row in rows]

    figure, ax = plt.subplots(figsize=(10, max(5, len(rows) * 0.48)))
    positions = list(range(len(rows)))
    ax.barh(positions, values, xerr=errors, capsize=3)
    ax.set_yticks(positions, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    if smaller_is_better:
        ax.invert_yaxis()
    for position, value in zip(positions, values):
        ax.text(value, position, f" {value:.4f}", va="center", fontsize=8)
    # figure.tight_layout()
    figure.tight_layout(
    pad=3.0,
    h_pad=3.0,
    w_pad=3.0
)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制RepairFormer消融结果图")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.summary))
    if not rows:
        raise ValueError("消融汇总CSV为空")
    output_dir = Path(args.output_dir)

    save_metric_plot(
        rows,
        "mpjpe_mean",
        "mpjpe_std",
        "Ablation study: mpjpe",
        "mpjpe (lower is better)",
        output_dir / "mpjpe.png",
        True,
    )
    save_metric_plot(
        rows,
        "pck@0.1_mean",
        "pck@0.1_std",
        "Ablation study: PCK@0.1",
        "PCK@0.1 (higher is better)",
        output_dir / "pck_at_0.1.png",
        False,
    )
    save_metric_plot(
        rows,
        "velocity_error_mean",
        "velocity_error_std",
        "Ablation study: velocity error",
        "Velocity error (lower is better)",
        output_dir / "velocity_error.png",
        True,
    )
    save_metric_plot(
        rows,
        "bone_error_mean",
        "bone_error_std",
        "Ablation study: bone error",
        "Bone error (lower is better)",
        output_dir / "bone_error.png",
        True,
    )
    if "mpjpe_delta_vs_full" in rows[0]:
        save_metric_plot(
            rows,
            "mpjpe_delta_vs_full",
            "mpjpe_std",
            "MPJPE change after removing each component",
            "Delta MPJPE vs full model",
            output_dir / "mpjpe_delta_vs_full.png",
            False,
        )
    print("消融图已保存：", output_dir.resolve())


if __name__ == "__main__":
    main()

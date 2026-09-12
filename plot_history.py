from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(path: Path):
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def plot_column(rows, x_key, keys, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = [int(float(row[x_key])) for row in rows]
    for key in keys:
        if key in rows[0]:
            ax.plot(x, [float(row[key]) for row in rows], label=key)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="绘制训练曲线")
    parser.add_argument("--history", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    history = Path(args.history)
    rows = read_history(history)
    if not rows:
        raise ValueError("history.csv 为空")
    output = Path(args.output_dir) if args.output_dir else history.parent / "curves"
    output.mkdir(parents=True, exist_ok=True)
    plot_column(rows, "epoch", ["train_loss", "val_loss"], output / "loss.png")
    plot_column(rows, "epoch", ["train_mpjpe", "val_mpjpe"], output / "mpjpe.png")
    # plot_column(rows, "epoch", ["train_gesture_acc", "val_gesture_acc"], output / "gesture_accuracy.png")
    print("曲线已保存:", output.resolve())


if __name__ == "__main__":
    main()

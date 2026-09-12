from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import load_config
from src.factory import make_dataset
from src.visualize import save_repair_animation, save_repair_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 DHG2016 数据格式并预览三维骨架")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-dir", default="dataset_preview")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    dataset = make_dataset(cfg, "all")
    print("数据摘要:", dataset.summary())
    sample = dataset[args.index % len(dataset)]
    x = sample["skeleton"].numpy()
    mask = np.ones(x.shape[:2], dtype=np.float32)
    masked = x.copy()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_repair_comparison(x, masked, x, mask, cfg["data"]["edges"], output / "skeleton_preview.png")
    save_repair_animation(x, masked, x, cfg["data"]["edges"], output / "skeleton_preview.gif")
    print("样本:", {k: v for k, v in sample.items() if k != "skeleton"})
    print("形状:", x.shape)
    print("已保存:", output.resolve())


if __name__ == "__main__":
    main()

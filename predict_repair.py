from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.factory import make_dataset, make_mask_generator, make_model
from src.utils import load_checkpoint, resolve_device
from src.visualize import save_repair_animation, save_repair_comparison


def main():
    parser = argparse.ArgumentParser(description="预览单个样本的骨架缺失修补过程")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mask-type", default="whole_finger")
    parser.add_argument("--ratio", type=float, default=0.30)
    parser.add_argument("--output-dir", default="repair_preview")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    device = resolve_device(cfg.get("device", "auto"))
    dataset = make_dataset(cfg, "test")
    sample = dataset[args.index % len(dataset)]
    x = sample["skeleton"].unsqueeze(0).to(device)
    generator = make_mask_generator(cfg)
    mask, _, _ = generator.generate(
        1, x.shape[1], x.shape[2], device,
        seed=cfg["eval"]["fixed_seed"], force_type=args.mask_type, force_ratio=args.ratio
    )
    model = make_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    with torch.no_grad():
        masked = x * mask.unsqueeze(-1)
        outputs = model(masked, mask)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    complete_np = x[0].cpu().numpy()
    masked_np = masked[0].cpu().numpy()
    repaired_np = outputs["repaired"][0].cpu().numpy()
    mask_np = mask[0].cpu().numpy()
    save_repair_comparison(
        complete_np, masked_np, repaired_np, mask_np,
        cfg["data"]["edges"], output / "repair_comparison.png"
    )
    save_repair_animation(
        complete_np, masked_np, repaired_np,
        cfg["data"]["edges"], output / "repair_animation.gif"
    )
    print("样本:", {k: v for k, v in sample.items() if k != "skeleton"})
    # print("分类预测 gesture/finger:",
    #       int(outputs["gesture_logits"].argmax(-1).item()) + 1,
    #       int(outputs["finger_logits"].argmax(-1).item()) + 1)
    print("预览已保存:", output.resolve())


if __name__ == "__main__":
    main()

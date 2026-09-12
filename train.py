from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import contextlib
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import load_config, save_config
from src.factory import make_loader, make_mask_generator, make_model
from src.losses import compute_losses
from src.metrics import missing_mpjpe, missing_velocity_error
from src.utils import append_csv, load_checkpoint, resolve_device, save_checkpoint, set_seed
from src.visualize import save_repair_comparison


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(
    model: torch.nn.Module,
    loader,
    mask_generator,
    cfg: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
    training = optimizer is not None
    model.train(training)

    totals: dict[str, float] = {}
    preview = None
    sample_count = 0
    amp_enabled = bool(cfg["train"].get("amp", True) and device.type == "cuda")
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))

    for batch_index, batch in enumerate(loader):
        skeleton = batch["skeleton"].to(device, non_blocking=True)
        batch_size = skeleton.size(0)
        seed = None if training else int(cfg["eval"]["fixed_seed"]) + batch_index

        visible_mask, _, _ = mask_generator.generate(
            batch_size=skeleton.size(0),
            num_frames=skeleton.size(1),
            num_joints=skeleton.size(2),
            device=device,
            seed=seed,
        )
        masked = skeleton * visible_mask.unsqueeze(-1)
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if amp_enabled else contextlib.nullcontext()
        )

        with torch.set_grad_enabled(training), autocast:
            outputs = model(masked, visible_mask)
            required = {"pred_coords", "pred_velocity", "repaired"}
            missing = required - outputs.keys()
            if missing:
                raise KeyError(f"模型输出缺少字段：{sorted(missing)}")

            loss, parts = compute_losses(
                outputs=outputs,
                target=skeleton,
                visible_mask=visible_mask,
                edges=cfg["data"]["edges"],
                loss_cfg=cfg["loss"],
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(f"batch={batch_index} 出现非有限损失：{loss.item()}")

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        metrics = {
            "loss": loss.item(),
            "mpjpe": missing_mpjpe(
                outputs["repaired"], skeleton, visible_mask
            ).item(),
            "velocity_error": missing_velocity_error(
                outputs["repaired"], skeleton, visible_mask
            ).item(),
            **{f"loss_{k}": v.item() for k, v in parts.items()},
        }

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        sample_count += batch_size

        if preview is None:
            preview = {
                "complete": skeleton[0].detach().cpu(),
                "masked": masked[0].detach().cpu(),
                "repaired": outputs["repaired"][0].detach().cpu(),
                "mask": visible_mask[0].detach().cpu(),
            }

    return {k: v / max(sample_count, 1) for k, v in totals.items()}, preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 DHG2016 纯三维骨架修补模型")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume",default="auto" ,help="纯修补检查点；auto 自动读取 latest.pt")
    parser.add_argument("--init-checkpoint", help="旧多任务检查点，仅加载修补主干")
    return parser.parse_args()


def prepare_training(cfg: dict[str, Any], device: torch.device):
    model = make_model(cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    epochs = int(cfg["train"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp_enabled = bool(cfg["train"].get("amp", True) and device.type == "cuda")
    scaler = make_grad_scaler(amp_enabled)
    return model, optimizer, scheduler, scaler, epochs


def save_preview(preview, cfg, path: Path) -> None:
    save_repair_comparison(
        preview["complete"].numpy(),
        preview["masked"].numpy(),
        preview["repaired"].numpy(),
        preview["mask"].numpy(),
        cfg["data"]["edges"],
        path,
    )


def main() -> None:
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume 与 --init-checkpoint 不能同时使用")

    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir

    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg.get("device", "auto"))
    output_dir = Path(cfg["train"]["output_dir"])
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "used_config.yaml")

    train_loader = make_loader(cfg, "train", shuffle=True)
    val_loader = make_loader(cfg, "val", shuffle=False)
    print("训练集：", train_loader.dataset.summary())
    print("验证集：", val_loader.dataset.summary())
    print("设备：", device)

    model, optimizer, scheduler, scaler, total_epochs = prepare_training(cfg, device)
    mask_generator = make_mask_generator(cfg)
    start_epoch, best_metric = 0, float("inf")

    resume = args.resume
    if resume == "auto":
        latest = output_dir / "last.pt"
        resume = str(latest) if latest.exists() else None

    if resume:
        loaded_epoch, best_metric, _ = load_checkpoint(
            resume, model, optimizer, scheduler, scaler, map_location=device
        )
        start_epoch = loaded_epoch + 1
        print(f"已恢复 {resume}，从 epoch {start_epoch} 继续训练")

    if args.init_checkpoint:
        load_checkpoint(args.init_checkpoint, model, map_location=device)
        start_epoch, best_metric = 0, float("inf")
        print(f"已加载旧主干：{args.init_checkpoint}")

    if start_epoch >= total_epochs:
        print("检查点已达到配置总轮数，无需继续训练")
        return

    patience = 0
    early_stop = int(cfg["train"]["early_stopping_patience"])
    save_every = int(cfg["train"]["save_every"])
    preview_every = int(cfg["train"]["preview_every"])

    for epoch in range(start_epoch, total_epochs):
        begin = time.time()
        train_metrics, _ = run_epoch(
            model, train_loader, mask_generator, cfg, device, optimizer, scaler
        )
        val_metrics, preview = run_epoch(
            model, val_loader, mask_generator, cfg, device
        )
        scheduler.step()

        elapsed = time.time() - begin
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        append_csv(output_dir / "history.csv", row)

        print(
            f"Epoch {epoch:03d} | train={train_metrics['loss']:.6f} "
            f"| val={val_metrics['loss']:.6f} "
            f"| MPJPE={val_metrics['mpjpe']:.6f} "
            f"| Velocity={val_metrics['velocity_error']:.6f} "
            f"| {elapsed:.1f}s"
        )

        improved = val_metrics["mpjpe"] < best_metric
        if improved:
            best_metric = val_metrics["mpjpe"]
            patience = 0
            save_checkpoint(
                output_dir / "best.pt",
                model, optimizer, scheduler, scaler,
                epoch, best_metric, cfg,
            )
            print("最佳模型保存")
        else:
            patience += 1

        save_checkpoint(
            output_dir / "last.pt",
            model, optimizer, scheduler, scaler,
            epoch, best_metric, cfg,
        )

        # if save_every > 0 and (epoch + 1) % save_every == 0:
        #     save_checkpoint(
        #         output_dir / f"epoch_{epoch + 1:03d}.pt",
        #         model, optimizer, scheduler, scaler,
        #         epoch, best_metric, cfg,
        #     )

        if preview and preview_every > 0 and (epoch + 1) % preview_every == 0:
            save_preview(preview, cfg, preview_dir / "epoch_newest.png")

        if early_stop > 0 and patience >= early_stop:
            print("验证集 MPJPE 长期未改善，提前停止")
            break

    print(f"训练结束，最佳验证集 MPJPE：{best_metric:.6f}")
    print("最佳模型：", (output_dir / "best.pt").resolve())


if __name__ == "__main__":
    main()

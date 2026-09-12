from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import contextlib
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.comparison_models import TRAINABLE_MODELS, build_repair_model
from src.config import load_config, save_config
from src.factory import make_loader, make_mask_generator, make_model
from src.losses import compute_losses
from src.metrics import missing_mpjpe, missing_velocity_error
from src.utils import load_checkpoint, resolve_device, save_checkpoint, set_seed


DEFAULT_CONFIG = "configs/comparison.yaml"
DEFAULT_PROPOSED_CONFIG = "configs/default.yaml"
DEFAULT_PROPOSED_CHECKPOINT = "runs/exp_dhg2016/best.pt"
DEFAULT_RUN_ROOT = "comparison_runs_six3"

# 无论 comparison_models.py 是否仍残留 repairformer，这里都会自动排除。
COMPARISON_MODELS = [name for name in TRAINABLE_MODELS if name != "repairformer"]


def configure_runtime() -> None:
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    result = dict(cfg["train"])
    result.update(cfg.get("comparison", {}).get("train", {}))
    return result


def loss_cfg(cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    result = dict(cfg["loss"])
    overrides = cfg.get("comparison", {}).get("loss_overrides", {})
    result.update(overrides.get(model_name, {}))
    return result


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_epoch(
    model: torch.nn.Module,
    loader,
    mask_generator,
    cfg: dict[str, Any],
    device: torch.device,
    current_loss_cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    current_train_cfg = train_cfg(cfg)
    amp_enabled = bool(current_train_cfg.get("amp", True) and device.type == "cuda")
    grad_clip = float(current_train_cfg.get("grad_clip", 0.0))
    totals: dict[str, float] = {}
    sample_count = 0

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
            loss, parts = compute_losses(
                outputs=outputs,
                target=skeleton,
                visible_mask=visible_mask,
                edges=cfg["data"]["edges"],
                loss_cfg=current_loss_cfg,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"batch={batch_index} 出现非有限损失：{loss.item()}"
            )

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
            **{f"loss_{key}": value.item() for key, value in parts.items()},
        }
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        sample_count += batch_size

    return {key: value / max(sample_count, 1) for key, value in totals.items()}


def train_one_model(
    model_name: str,
    cfg: dict[str, Any],
    run_root: Path,
    device: torch.device,
    skip_existing: bool,
    resume_existing: bool,
) -> Path:
    model_dir = run_root / model_name
    best_path = model_dir / "best.pt"
    latest_path = model_dir / "latest.pt"

    if skip_existing and best_path.exists():
        print(f"[{model_name}] 已存在 best.pt，跳过训练")
        return best_path

    local_cfg = copy.deepcopy(cfg)
    local_cfg["train"].update(local_cfg.get("comparison", {}).get("train", {}))
    local_cfg["train"]["output_dir"] = str(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_config(local_cfg, model_dir / "used_config.yaml")

    set_seed(int(local_cfg["seed"]))
    train_loader = make_loader(local_cfg, "train", shuffle=True)
    val_loader = make_loader(local_cfg, "val", shuffle=False)
    mask_generator = make_mask_generator(local_cfg)
    model = build_repair_model(model_name, local_cfg).to(device)

    current_train_cfg = train_cfg(local_cfg)
    optimizer = AdamW(
        model.parameters(),
        lr=float(current_train_cfg["learning_rate"]),
        weight_decay=float(current_train_cfg["weight_decay"]),
    )
    epochs = int(current_train_cfg["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp_enabled = bool(current_train_cfg.get("amp", True) and device.type == "cuda")
    scaler = make_grad_scaler(amp_enabled)
    current_loss_cfg = loss_cfg(local_cfg, model_name)

    start_epoch, best_metric = 0, float("inf")
    if resume_existing and latest_path.exists():
        loaded_epoch, best_metric, _ = load_checkpoint(
            latest_path, model, optimizer, scheduler, scaler, map_location=device
        )
        start_epoch = loaded_epoch + 1
    elif (model_dir / "history.csv").exists():
        (model_dir / "history.csv").unlink()

    patience = 0
    early_stop = int(current_train_cfg.get("early_stopping_patience", 0))
    save_every = int(current_train_cfg.get("save_every", 10))

    print(f"\n{'=' * 72}\n训练模型：{model_name}\n{'=' * 72}")
    for epoch in range(start_epoch, epochs):
        begin = time.time()
        train_metrics = run_epoch(
            model, train_loader, mask_generator, local_cfg, device,
            current_loss_cfg, optimizer, scaler
        )
        val_metrics = run_epoch(
            model, val_loader, mask_generator, local_cfg, device, current_loss_cfg
        )
        scheduler.step()
        elapsed = time.time() - begin

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        append_row(model_dir / "history.csv", row)
        print(
            f"[{model_name}] Epoch {epoch:03d} "
            f"| train={train_metrics['loss']:.6f} "
            f"| val={val_metrics['loss']:.6f} "
            f"| MPJPE={val_metrics['mpjpe']:.6f} "
            f"| {elapsed:.1f}s"
        )

        current = val_metrics["mpjpe"]
        if current < best_metric:
            best_metric, patience = current, 0
            save_checkpoint(
                best_path, model, optimizer, scheduler, scaler,
                epoch, best_metric, local_cfg
            )
        else:
            patience += 1

        save_checkpoint(
            latest_path, model, optimizer, scheduler, scaler,
            epoch, best_metric, local_cfg
        )
        # if save_every > 0 and (epoch + 1) % save_every == 0:
        #     save_checkpoint(
        #         model_dir / f"epoch_{epoch + 1:03d}.pt",
        #         model, optimizer, scheduler, scaler,
        #         epoch, best_metric, local_cfg
        #     )
        if early_stop > 0 and patience >= early_stop:
            print(f"[{model_name}] 验证集长期未改善，提前停止")
            break

    if not best_path.exists():
        raise FileNotFoundError(f"{model_name} 未生成最佳权重：{best_path}")
    return best_path


def bone_error(
    repaired: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    edges: list[list[int]],
) -> torch.Tensor:
    missing = visible_mask < 0.5
    errors, valid_masks = [], []

    for i, j in edges:
        i, j = int(i), int(j)
        pred_length = torch.linalg.vector_norm(
            repaired[:, :, i] - repaired[:, :, j], dim=-1
        )
        true_length = torch.linalg.vector_norm(
            target[:, :, i] - target[:, :, j], dim=-1
        )
        errors.append((pred_length - true_length).abs())
        valid_masks.append(missing[:, :, i] | missing[:, :, j])

    all_errors = torch.stack(errors, dim=-1)
    all_valid = torch.stack(valid_masks, dim=-1)
    return all_errors[all_valid].mean() if all_valid.any() else repaired.new_zeros(())


def missing_pck(
    repaired: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    distances = torch.linalg.vector_norm(repaired - target, dim=-1)
    missing = visible_mask < 0.5
    return (
        (distances[missing] <= threshold).float().mean()
        if missing.any() else repaired.new_zeros(())
    )


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    mask_generator,
    cfg: dict[str, Any],
    device: torch.device,
    method_name: str,
) -> list[dict[str, Any]]:
    model.eval()
    thresholds = [
        float(value)
        for value in cfg["eval"].get("pck_thresholds", [0.1, 0.2, 0.5])
    ]
    mask_types = cfg["eval"].get("mask_types", [None])
    ratios = [float(value) for value in cfg["eval"].get("ratios", [0.3])]
    rows: list[dict[str, Any]] = []

    for mask_type in mask_types:
        for ratio in ratios:
            totals = {
                "mpjpe": 0.0,
                "velocity_error": 0.0,
                "bone_error": 0.0,
                "inference_ms": 0.0,
                **{f"pck@{value:g}": 0.0 for value in thresholds},
            }
            sample_count, warmed_up = 0, False

            for batch_index, batch in enumerate(loader):
                skeleton = batch["skeleton"].to(device, non_blocking=True)
                batch_size = skeleton.size(0)
                visible_mask, _, _ = mask_generator.generate(
                    batch_size=skeleton.size(0),
                    num_frames=skeleton.size(1),
                    num_joints=skeleton.size(2),
                    device=device,
                    seed=int(cfg["eval"]["fixed_seed"]) + batch_index,
                    force_type=mask_type,
                    force_ratio=ratio,
                )
                masked = skeleton * visible_mask.unsqueeze(-1)

                if not warmed_up:
                    _ = model(masked, visible_mask)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    warmed_up = True

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                begin = time.perf_counter()
                outputs = model(masked, visible_mask)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - begin) * 1000.0

                repaired = outputs["repaired"]
                values = {
                    "mpjpe": missing_mpjpe(
                        repaired, skeleton, visible_mask
                    ).item(),
                    "velocity_error": missing_velocity_error(
                        repaired, skeleton, visible_mask
                    ).item(),
                    "bone_error": bone_error(
                        repaired, skeleton, visible_mask, cfg["data"]["edges"]
                    ).item(),
                    "inference_ms": elapsed_ms / batch_size,
                    **{
                        f"pck@{value:g}": missing_pck(
                            repaired, skeleton, visible_mask, value
                        ).item()
                        for value in thresholds
                    },
                }
                for key, value in values.items():
                    totals[key] += value * batch_size
                sample_count += batch_size

            row = {
                "method": method_name,
                "mask_type": mask_type or "random",
                "ratio": ratio,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                **{
                    key: value / max(sample_count, 1)
                    for key, value in totals.items()
                },
            }
            rows.append(row)
            print(
                f"[测试] {method_name} | {row['mask_type']} | ratio={ratio:.2f} "
                f"| MPJPE={row['mpjpe']:.6f}"
            )
    return rows


def load_comparison_model(
    name: str,
    checkpoint: Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    model = build_repair_model(name, cfg).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    return model


def load_proposed_model(
    checkpoint: Path,
    proposed_cfg: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    model = make_model(proposed_cfg).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    return model


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    metric_keys = [
        key for key in rows[0]
        if key not in {"method", "mask_type", "ratio", "parameters"}
    ]
    return [
        {
            "method": method,
            "parameters": next(
                row["parameters"] for row in rows if row["method"] == method
            ),
            **{
                key: sum(
                    float(row[key]) for row in rows if row["method"] == method
                ) / sum(row["method"] == method for row in rows)
                for key in metric_keys
            },
        }
        for method in methods
    ]


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(result_dir: Path, summary: list[dict[str, Any]]) -> None:
    methods = [row["method"] for row in summary]
    for metric in ("mpjpe", "velocity_error", "bone_error", "inference_ms"):
        values = [float(row[metric]) for row in summary]
        plt.figure(figsize=(9, 5))
        bars = plt.barh(methods, values)
        plt.xlabel(metric)
        plt.title(f"Model comparison: {metric}")

        # ====== 新增：条形上标注数值 ======
        # 保留3位小数；如需2位改为 "%.2f"
        plt.bar_label(bars, fmt="%.4f", padding=6, fontsize=10)
        # 拓宽X轴，防止数字被截断
        plt.xlim(right=max(values) * 1.13)

        plt.tight_layout()
        plt.savefig(result_dir / f"{metric}.png", dpi=180, bbox_inches="tight")
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练六个对比修补模型，再加入提出模型权重统一测试"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--proposed-config", default=DEFAULT_PROPOSED_CONFIG)
    parser.add_argument("--data-root")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--proposed-checkpoint", default=DEFAULT_PROPOSED_CHECKPOINT
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=COMPARISON_MODELS,
        choices=COMPARISON_MODELS,
    )
   
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime()

    cfg = load_config(args.config)
    proposed_cfg = load_config(args.proposed_config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
        proposed_cfg["data"]["root"] = args.data_root

    device = resolve_device(cfg.get("device", "auto"))
    run_root = Path(args.run_root)
    result_dir = run_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    proposed_checkpoint = Path(args.proposed_checkpoint)

    print("设备：", device)
    print("对比模型：", args.models)
    print("提出模型配置：", Path(args.proposed_config).resolve())
    print("提出模型权重：", proposed_checkpoint.resolve())

    checkpoints: dict[str, Path] = {}
    if args.test_only:
        for name in args.models:
            checkpoint = run_root / name / "best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"找不到对比模型权重：{checkpoint}")
            checkpoints[name] = checkpoint
    else:
        for name in args.models:
            checkpoints[name] = train_one_model(
                name, cfg, run_root, device,
                args.skip_existing, args.resume_existing
            )

    if not proposed_checkpoint.exists():
        raise FileNotFoundError(
            f"找不到提出模型权重：{proposed_checkpoint.resolve()}"
        )

    test_loader = make_loader(cfg, "test", shuffle=False)
    mask_generator = make_mask_generator(cfg)
    detailed_rows: list[dict[str, Any]] = []

    for name, checkpoint in checkpoints.items():
        model = load_comparison_model(name, checkpoint, cfg, device)
        detailed_rows.extend(
            evaluate_model(model, test_loader, mask_generator, cfg, device, name)
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    proposed_model = load_proposed_model(
        proposed_checkpoint, proposed_cfg, device
    )
    detailed_rows.extend(
        evaluate_model(
            proposed_model, test_loader, mask_generator, cfg,
            device, "AGTF"
        )
    )

    summary_rows = summarize(detailed_rows)
    save_csv(result_dir / "detailed_results.csv", detailed_rows)
    save_csv(result_dir / "summary_results.csv", summary_rows)
    with (result_dir / "summary_results.json").open("w", encoding="utf-8") as file:
        json.dump(summary_rows, file, ensure_ascii=False, indent=2)
    save_plots(result_dir, summary_rows)

    print("\n统一测试完成")
    print("详细结果：", (result_dir / "detailed_results.csv").resolve())
    print("汇总结果：", (result_dir / "summary_results.csv").resolve())
    print("图表目录：", result_dir.resolve())


if __name__ == "__main__":
    main()

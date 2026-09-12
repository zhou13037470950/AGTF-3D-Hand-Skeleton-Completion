from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.ablation_models import (
    ABLATION_SPECS,
    build_ablation_model,
    get_ablation_spec,
    loss_config_for_variant,
    trainable_parameter_count,
)
from src.config import load_config
from src.factory import make_loader, make_mask_generator
from src.losses import compute_losses
from src.metrics import missing_mpjpe, missing_velocity_error
from src.utils import resolve_device, set_seed


# ============================================================================
# F5 默认参数区：不传命令行参数时直接使用这些值
# ============================================================================
DEFAULT_BASE_CONFIG = "configs/default.yaml"
DEFAULT_EXPERIMENT_CONFIG = "configs/ablation_experiment.yaml"
DEFAULT_MODE = "all"  # train / evaluate / all


def configure_runtime() -> None:
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML 顶层必须是字典：{path}")
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_metric: float,
    variant: str,
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    state = model.state_dict()
    return {
        "model": state,
        "model_state_dict": state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "variant": variant,
        "seed": int(seed),
        "config": cfg,
    }


def save_checkpoint(path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(**kwargs), path)


def torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model_state(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch_load(path, device)
    state = checkpoint.get(
        "model",
        checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)),
    )
    if not isinstance(state, dict):
        raise TypeError(f"检查点中找不到模型权重：{path}")
    model.load_state_dict(state, strict=strict)
    return checkpoint


def restore_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = load_model_state(path, model, device, strict=True)
    optimizer_state = checkpoint.get(
        "optimizer", checkpoint.get("optimizer_state_dict")
    )
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", -1)) + 1, float(
        checkpoint.get("best_metric", math.inf)
    )


def missing_pck(
    repaired: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(repaired - target, dim=-1)
    missing = visible_mask < 0.5
    if not missing.any():
        return repaired.new_zeros(())
    return (distance[missing] <= threshold).float().mean()


def missing_bone_error(
    repaired: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    edges: Iterable[Iterable[int]],
) -> torch.Tensor:
    missing = visible_mask < 0.5
    errors: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    for start, end in edges:
        start, end = int(start), int(end)
        pred_length = torch.linalg.vector_norm(
            repaired[:, :, start] - repaired[:, :, end], dim=-1
        )
        true_length = torch.linalg.vector_norm(
            target[:, :, start] - target[:, :, end], dim=-1
        )
        errors.append((pred_length - true_length).abs())
        valid.append(missing[:, :, start] | missing[:, :, end])
    if not errors:
        return repaired.new_zeros(())
    errors_tensor = torch.stack(errors, dim=-1)
    valid_tensor = torch.stack(valid, dim=-1)
    return (
        errors_tensor[valid_tensor].mean()
        if valid_tensor.any()
        else repaired.new_zeros(())
    )


def run_train_epoch(
    *,
    model: torch.nn.Module,
    loader: Any,
    mask_generator: Any,
    cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    amp_enabled = bool(cfg["train"].get("amp", True) and device.type == "cuda")
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))
    totals: dict[str, float] = {}
    sample_count = 0

    for batch_index, batch in enumerate(loader):
        skeleton = batch["skeleton"].to(device, non_blocking=True)
        batch_size = int(skeleton.shape[0])
        visible_mask, _, _ = mask_generator.generate(
            batch_size=batch_size,
            num_frames=skeleton.shape[1],
            num_joints=skeleton.shape[2],
            device=device,
            seed=cfg["seed"] + epoch + batch_index, # 训练阶段固定掩码序列
        )
        masked = skeleton * visible_mask.unsqueeze(-1)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            outputs = model(masked, visible_mask)
            loss, parts = compute_losses(
                outputs=outputs,
                target=skeleton,
                visible_mask=visible_mask,
                edges=cfg["data"]["edges"],
                loss_cfg=loss_cfg,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"batch={batch_index} 出现非有限损失：{float(loss.detach())}"
            )

        if amp_enabled:
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
            "loss": float(loss.detach()),
            "mpjpe": float(
                missing_mpjpe(outputs["repaired"], skeleton, visible_mask)
            ),
            "velocity_error": float(
                missing_velocity_error(outputs["repaired"], skeleton, visible_mask)
            ),
            **{f"loss_{key}": float(value.detach()) for key, value in parts.items()},
        }
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        sample_count += batch_size

    return {key: value / max(sample_count, 1) for key, value in totals.items()}


@torch.inference_mode()
def evaluate_grid(
    *,
    model: torch.nn.Module,
    loader: Any,
    mask_generator: Any,
    cfg: dict[str, Any],
    device: torch.device,
    split: str,
    variant: str,
    seed: int,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    mask_types = list(cfg["eval"]["mask_types"])
    ratios = [float(value) for value in cfg["eval"]["ratios"]]
    fixed_seed = int(cfg["eval"].get("fixed_seed", 1021))

    for mask_type in mask_types:
        for ratio in ratios:
            totals = {
                "mpjpe": 0.0,
                "velocity_error": 0.0,
                "bone_error": 0.0,
                "pck@0.1": 0.0,
            }
            sample_count = 0
            for batch_index, batch in enumerate(loader):
                skeleton = batch["skeleton"].to(device, non_blocking=True)
                batch_size = int(skeleton.shape[0])
                visible_mask, _, _ = mask_generator.generate(
                    batch_size=batch_size,
                    num_frames=skeleton.shape[1],
                    num_joints=skeleton.shape[2],
                    device=device,
                    seed=fixed_seed + batch_index,
                    force_type=mask_type,
                    force_ratio=ratio,
                )
                masked = skeleton * visible_mask.unsqueeze(-1)
                outputs = model(masked, visible_mask)
                repaired = outputs["repaired"]
                values = {
                    "mpjpe": float(missing_mpjpe(repaired, skeleton, visible_mask)),
                    "velocity_error": float(
                        missing_velocity_error(repaired, skeleton, visible_mask)
                    ),
                    "bone_error": float(
                        missing_bone_error(
                            repaired,
                            skeleton,
                            visible_mask,
                            cfg["data"]["edges"],
                        )
                    ),
                    "pck@0.1": float(
                        missing_pck(repaired, skeleton, visible_mask, 0.1)
                    ),
                }
                for key, value in values.items():
                    totals[key] += value * batch_size
                sample_count += batch_size

            rows.append(
                {
                    "split": split,
                    "variant": variant,
                    "seed": seed,
                    "mask_type": mask_type,
                    "ratio": ratio,
                    **{
                        key: value / max(sample_count, 1)
                        for key, value in totals.items()
                    },
                }
            )
    return rows


def summarize_grid(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_keys = ("mpjpe", "velocity_error", "bone_error", "pck@0.1")
    return {
        key: statistics.fmean(float(row[key]) for row in rows)
        for key in metric_keys
    }


@torch.inference_mode()
def measure_inference_ms(
    *,
    model: torch.nn.Module,
    loader: Any,
    mask_generator: Any,
    cfg: dict[str, Any],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    model.eval()
    batch = next(iter(loader))
    skeleton = batch["skeleton"].to(device, non_blocking=True)
    visible_mask, _, _ = mask_generator.generate(
        batch_size=skeleton.shape[0],
        num_frames=skeleton.shape[1],
        num_joints=skeleton.shape[2],
        device=device,
        seed=int(cfg["eval"].get("fixed_seed", 1021)),
        force_type="random_joint",
        force_ratio=0.3,
    )
    masked = skeleton * visible_mask.unsqueeze(-1)

    for _ in range(max(0, warmup)):
        model(masked, visible_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    measurements: list[float] = []
    for _ in range(max(1, repeats)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        model(masked, visible_mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        measurements.append(
            (time.perf_counter() - begin) * 1000.0 / skeleton.shape[0]
        )
    return float(statistics.median(measurements))


def experiment_settings(
    base_cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
) -> tuple[list[str], list[str], list[int], Path, dict[str, Any]]:
    section = exp_cfg.get("ablation", exp_cfg)

    # 训练列表与测试列表分开：完整模型已有权重时，不必再次训练，
    # 但测试列表仍保留 full，作为所有消融变体的基准。
    train_variants = [
        str(value)
        for value in section.get("train_variants", section.get("variants", []))
    ]
    evaluate_variants = [
        str(value)
        for value in section.get(
            "evaluate_variants",
            section.get("variants", ["full"]),
        )
    ]

    for variant in dict.fromkeys([*train_variants, *evaluate_variants]):
        get_ablation_spec(variant)

    if section.get("seeds"):
        seeds = [int(value) for value in section["seeds"]]
    else:
        repeats = int(section.get("repeats", 1))
        start_seed = int(section.get("start_seed", base_cfg.get("seed", 42)))
        seeds = [start_seed + index for index in range(repeats)]

    output_root = Path(section.get("output_root", "runs/ablation"))
    train_overrides = dict(section.get("train", {}))
    return train_variants, evaluate_variants, seeds, output_root, train_overrides


def resolve_checkpoint(
    *,
    section: dict[str, Any],
    variant: str,
    seed: int,
    output_root: Path,
) -> Path:
    """优先读取 YAML 中指定的已有权重，否则使用默认消融目录。"""
    configured = section.get("checkpoints", {}).get(variant)

    # 一个变体只指定一份权重，例如 full。
    if isinstance(configured, str) and configured.strip():
        return Path(configured)

    # 一个变体按随机种子指定多份权重。
    if isinstance(configured, dict):
        value = configured.get(str(seed), configured.get(seed))
        if value is None:
            value = configured.get("default")
        if value:
            return Path(value)

    return output_root / variant / f"seed_{seed}" / "best.pt"

def train_one_run(
    *,
    variant: str,
    seed: int,
    base_cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
    output_root: Path,
    train_overrides: dict[str, Any],
    device: torch.device,
    skip_existing: bool,
    resume: bool,
) -> Path:
    run_dir = output_root / variant / f"seed_{seed}"
    best_path = run_dir / "best.pt"
    latest_path = run_dir / "latest.pt"
    if skip_existing and best_path.exists():
        print(f"[{variant} seed={seed}] 已存在 best.pt，跳过")
        return best_path

    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["train"].update(train_overrides)
    cfg["train"]["output_dir"] = str(run_dir)
    cfg["ablation_variant"] = variant
    cfg["ablation_description"] = get_ablation_spec(variant).description
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(run_dir / "used_config.yaml", cfg)
    save_yaml(
        run_dir / "variant.yaml",
        {
            "name": variant,
            "label": get_ablation_spec(variant).label,
            "description": get_ablation_spec(variant).description,
            "loss": loss_config_for_variant(cfg["loss"], variant),
        },
    )

    set_seed(seed)
    train_loader = make_loader(cfg, "train", shuffle=True)
    val_loader = make_loader(cfg, "val", shuffle=False)
    mask_generator = make_mask_generator(cfg)
    model = build_ablation_model(cfg, variant).to(device)
    loss_cfg = loss_config_for_variant(cfg["loss"], variant)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    epochs = int(cfg["train"]["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp_enabled = bool(cfg["train"].get("amp", True) and device.type == "cuda")
    scaler = make_grad_scaler(amp_enabled)

    start_epoch, best_metric = 0, math.inf
    if resume and latest_path.exists():
        start_epoch, best_metric = restore_training_state(
            latest_path,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        print(f"[{variant} seed={seed}] 从 epoch={start_epoch} 继续")
    elif (run_dir / "history.csv").exists():
        (run_dir / "history.csv").unlink()

    section = exp_cfg.get("ablation", exp_cfg)
    grid_every = int(section.get("full_grid_val_every", 5))
    patience_limit = int(cfg["train"].get("early_stopping_patience", 0))
    save_every = int(cfg["train"].get("save_every", 10))
    patience = 0

    print("\n" + "=" * 80)
    print(
        f"训练消融：{variant} | {get_ablation_spec(variant).label} | "
        f"seed={seed} | params={trainable_parameter_count(model):,}"
    )
    print("=" * 80)

    for epoch in range(start_epoch, epochs):
        begin = time.time()
        train_metrics = run_train_epoch(
            model=model,
            loader=train_loader,
            mask_generator=mask_generator,
            cfg=cfg,
            loss_cfg=loss_cfg,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch
        )
        scheduler.step()

        do_grid = (
            epoch == 0
            or epoch + 1 == epochs
            or (grid_every > 0 and (epoch + 1) % grid_every == 0)
        )
        grid_summary: dict[str, float] = {}
        if do_grid:
            grid_rows = evaluate_grid(
                model=model,
                loader=val_loader,
                mask_generator=mask_generator,
                cfg=cfg,
                device=device,
                split="val",
                variant=variant,
                seed=seed,
            )
            grid_summary = summarize_grid(grid_rows)
            write_csv(run_dir / f"val_grid_epoch_{epoch + 1:03d}.csv", grid_rows)

            current = grid_summary["mpjpe"]
            if current < best_metric:
                best_metric, patience = current, 0
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                    variant=variant,
                    seed=seed,
                    cfg=cfg,
                )
            else:
                patience += 1

        elapsed = time.time() - begin
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_grid_{key}": value for key, value in grid_summary.items()},
        }
        append_csv(run_dir / "history.csv", row)
        grid_text = (
            f" | grid_MPJPE={grid_summary['mpjpe']:.6f}"
            f" | grid_PCK01={grid_summary['pck@0.1']:.6f}"
            if grid_summary
            else ""
        )
        print(
            f"[{variant} seed={seed}] Epoch {epoch + 1:03d} "
            f"| train={train_metrics['loss']:.6f}"
            f"{grid_text} | {elapsed:.1f}s"
        )

        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_metric=best_metric,
            variant=variant,
            seed=seed,
            cfg=cfg,
        )
        if save_every > 0 and (epoch + 1) % save_every == 0:
            save_checkpoint(
                run_dir / f"epoch_{epoch + 1:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                variant=variant,
                seed=seed,
                cfg=cfg,
            )

        if (
            patience_limit > 0
            and do_grid
            and patience >= patience_limit
        ):
            print(f"[{variant} seed={seed}] 完整验证长期未改善，提前停止")
            break

    if not best_path.exists():
        raise FileNotFoundError(f"未生成 best.pt：{best_path}")
    return best_path


def evaluate_all(
    *,
    variants: list[str],
    seeds: list[int],
    base_cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
    output_root: Path,
    device: torch.device,
) -> None:
    detail_rows: list[dict[str, Any]] = []
    run_summary_rows: list[dict[str, Any]] = []
    section = exp_cfg.get("ablation", exp_cfg)
    timing_warmup = int(section.get("timing_warmup", 20))
    timing_repeats = int(section.get("timing_repeats", 50))

    for variant in variants:
        for seed in seeds:
            run_dir = output_root / variant / f"seed_{seed}"
            checkpoint = resolve_checkpoint(
                section=section,
                variant=variant,
                seed=seed,
                output_root=output_root,
            )
            if not checkpoint.exists():
                print(f"[跳过测试] 找不到权重：{checkpoint}")
                continue

            print(f"[加载权重] {variant} seed={seed}: {checkpoint}")

            cfg_path = run_dir / "used_config.yaml"
            cfg = load_yaml(cfg_path) if cfg_path.exists() else copy.deepcopy(base_cfg)
            set_seed(seed)
            test_loader = make_loader(cfg, "test", shuffle=False)
            mask_generator = make_mask_generator(cfg)
            model = build_ablation_model(cfg, variant).to(device)
            load_model_state(checkpoint, model, device, strict=True)

            rows = evaluate_grid(
                model=model,
                loader=test_loader,
                mask_generator=mask_generator,
                cfg=cfg,
                device=device,
                split="test",
                variant=variant,
                seed=seed,
            )
            latency = measure_inference_ms(
                model=model,
                loader=test_loader,
                mask_generator=mask_generator,
                cfg=cfg,
                device=device,
                warmup=timing_warmup,
                repeats=timing_repeats,
            )
            parameters = trainable_parameter_count(model)
            for row in rows:
                row["parameters"] = parameters
                row["inference_ms"] = latency
            detail_rows.extend(rows)
            summary = summarize_grid(rows)
            run_summary_rows.append(
                {
                    "variant": variant,
                    "label": get_ablation_spec(variant).label,
                    "seed": seed,
                    "parameters": parameters,
                    "inference_ms": latency,
                    **summary,
                }
            )
            write_csv(run_dir / "test_detail.csv", rows)
            save_json(run_dir / "test_detail.json", rows)
            print(
                f"[测试完成] {variant} seed={seed} "
                f"MPJPE={summary['mpjpe']:.6f} "
                f"PCK@0.1={summary['pck@0.1']:.6f}"
            )

    result_dir = output_root / "results"
    write_csv(result_dir / "ablation_detail.csv", detail_rows)
    write_csv(result_dir / "ablation_runs.csv", run_summary_rows)
    save_json(result_dir / "ablation_detail.json", detail_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for variant in variants:
        selected = [row for row in run_summary_rows if row["variant"] == variant]
        if not selected:
            continue
        aggregate: dict[str, Any] = {
            "variant": variant,
            "label": get_ablation_spec(variant).label,
            "runs": len(selected),
            "parameters": int(statistics.fmean(row["parameters"] for row in selected)),
        }
        for metric in (
            "mpjpe",
            "velocity_error",
            "bone_error",
            "pck@0.1",
            "inference_ms",
        ):
            values = [float(row[metric]) for row in selected]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregate_rows.append(aggregate)

    full_row = next(
        (row for row in aggregate_rows if row["variant"] == "full"), None
    )
    if full_row:
        for row in aggregate_rows:
            row["mpjpe_delta_vs_full"] = (
                row["mpjpe_mean"] - full_row["mpjpe_mean"]
            )
            row["pck01_delta_vs_full"] = (
                row["pck@0.1_mean"] - full_row["pck@0.1_mean"]
            )

    write_csv(result_dir / "ablation_summary.csv", aggregate_rows)
    save_json(result_dir / "ablation_summary.json", aggregate_rows)
    print(f"消融汇总已保存：{result_dir.resolve()}")


def parse_names(value: str | None, defaults: list[str]) -> list[str]:
    if value is None or value.strip().lower() == "all":
        return defaults
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RepairFormer 结构/损失消融：训练未训练的变体，并读取已有权重统一测试"
    )
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--experiment-config", default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "all"],
        default=DEFAULT_MODE,
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="兼容参数：指定后同时覆盖训练和测试列表",
    )
    parser.add_argument(
        "--train-variants",
        default=None,
        help="逗号分隔；默认读取 YAML 的 train_variants",
    )
    parser.add_argument(
        "--evaluate-variants",
        default=None,
        help="逗号分隔；默认读取 YAML 的 evaluate_variants",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--skip-existing", default=True,action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_runtime()
    args = parse_args()
    base_cfg = load_config(args.base_config)
    exp_cfg = load_yaml(args.experiment_config)
    if args.data_root:
        base_cfg["data"]["root"] = args.data_root

    (
        default_train_variants,
        default_evaluate_variants,
        seeds,
        output_root,
        train_overrides,
    ) = experiment_settings(base_cfg, exp_cfg)

    if args.variants is not None:
        train_variants = parse_names(args.variants, default_train_variants)
        evaluate_variants = parse_names(args.variants, default_evaluate_variants)
    else:
        train_variants = parse_names(
            args.train_variants,
            default_train_variants,
        )
        evaluate_variants = parse_names(
            args.evaluate_variants,
            default_evaluate_variants,
        )

    for variant in dict.fromkeys([*train_variants, *evaluate_variants]):
        get_ablation_spec(variant)

    device = resolve_device(base_cfg.get("device", "auto"))
    print("设备：", device)
    print("训练变体：", train_variants)
    print("测试变体：", evaluate_variants)
    print("重复种子：", seeds)

    if args.mode in {"train", "all"}:
        if not train_variants:
            print("训练列表为空，跳过训练。")
        for variant in train_variants:
            for seed in seeds:
                train_one_run(
                    variant=variant,
                    seed=seed,
                    base_cfg=base_cfg,
                    exp_cfg=exp_cfg,
                    output_root=output_root,
                    train_overrides=train_overrides,
                    device=device,
                    skip_existing=args.skip_existing,
                    resume=args.resume,
                )

    if args.mode in {"evaluate", "all"}:
        evaluate_all(
            variants=evaluate_variants,
            seeds=seeds,
            base_cfg=base_cfg,
            exp_cfg=exp_cfg,
            output_root=output_root,
            device=device,
        )


if __name__ == "__main__":
    main()

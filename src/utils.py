from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    return torch.device(name)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "scaler": (
                scaler.state_dict()
                if scaler is not None
                else None
            ),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, float, dict[str, Any]]:
    """
    加载检查点，并兼容旧模型中的分类头参数。
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"检查点文件不存在：{path.resolve()}"
        )

    print(f"\n正在加载检查点：{path}")

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if "model" not in checkpoint:
        raise KeyError(
            f"检查点中不存在'model'字段：{path}"
        )

    checkpoint_state = checkpoint["model"]
    current_state = model.state_dict()

    ignored_prefixes = (
        "gesture_head.",
        "finger_head.",
    )

    filtered_state: dict[str, torch.Tensor] = {}
    ignored_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatch_keys = []

    for original_key, value in checkpoint_state.items():
        key = original_key

        if key.startswith("module."):
            key = key[len("module."):]

        if key.startswith(ignored_prefixes):
            ignored_keys.append(key)
            continue

        if key not in current_state:
            unexpected_keys.append(key)
            continue

        if current_state[key].shape != value.shape:
            shape_mismatch_keys.append(
                (
                    key,
                    tuple(value.shape),
                    tuple(current_state[key].shape),
                )
            )
            continue

        filtered_state[key] = value

    load_result = model.load_state_dict(
        filtered_state,
        strict=False,
    )

    missing_core_keys = [
        key
        for key in load_result.missing_keys
        if not key.startswith(ignored_prefixes)
    ]

    print("\n检查点加载情况：")
    print(f"成功加载：{len(filtered_state)}个参数")


    if ignored_keys:
        for key in ignored_keys:
            print(f"  忽略：{key}")

    errors = []

    if unexpected_keys:
        errors.append(
            "无法识别的参数：\n"
            + "\n".join(
                f"  - {key}"
                for key in unexpected_keys
            )
        )

    if shape_mismatch_keys:
        mismatch_text = []

        for key, old_shape, new_shape in shape_mismatch_keys:
            mismatch_text.append(
                f"  - {key}: "
                f"checkpoint={old_shape}, "
                f"model={new_shape}"
            )

        errors.append(
            "形状不匹配的参数：\n"
            + "\n".join(mismatch_text)
        )

    if missing_core_keys:
        errors.append(
            "当前模型缺少的核心参数：\n"
            + "\n".join(
                f"  - {key}"
                for key in missing_core_keys
            )
        )

    if errors:
        raise RuntimeError(
            "检查点与当前模型的核心结构不一致：\n\n"
            + "\n\n".join(errors)
        )

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )
        except (ValueError, RuntimeError) as error:
            print(
                "警告：优化器状态无法恢复，"
                "将使用新的优化器状态。\n"
                f"原因：{error}"
            )

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(
                checkpoint["scheduler"]
            )
        except (ValueError, RuntimeError) as error:
            print(
                f"警告：调度器状态无法恢复：{error}"
            )

    if scaler is not None and checkpoint.get("scaler") is not None:
        try:
            scaler.load_state_dict(
                checkpoint["scaler"]
            )
        except (ValueError, RuntimeError) as error:
            print(
                f"警告：Scaler状态无法恢复：{error}"
            )

    return (
        int(checkpoint.get("epoch", -1)),
        float(
            checkpoint.get(
                "best_metric",
                float("inf"),
            )
        ),
        checkpoint,
    )


def append_csv(
    path: str | Path,
    row: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = not path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def save_json(
    path: str | Path,
    value: Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.ablation_models import build_ablation_model
from src.config import load_config
from src.factory import make_dataset, make_mask_generator, make_model
from src.utils import resolve_device, set_seed

try:
    from src.comparison_models import build_repair_model
except ImportError:
    build_repair_model = None


# ============================================================================
# F5 默认参数区
# ============================================================================
DEFAULT_MANIFEST = "configs/visualization_models.yaml"
DEFAULT_OUTPUT_DIR = "runs/model_visualization"

# 三类关节使用固定且明显区分的颜色
GROUND_TRUTH_COLOR = "#2CA02C"  # 真实/可见关节：绿色
MISSING_COLOR = "#D62728"       # 缺失关节：红色
REPAIRED_COLOR = "#1F77B4"      # 修补关节：蓝色
BONE_COLOR = "#555555"          # 骨架连线：灰色
ERROR_COLOR = "#9467BD"         # 预测误差线：紫色


@dataclass
class LoadedModel:
    key: str
    label: str
    model: torch.nn.Module
    checkpoint: Path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise TypeError(f"YAML 顶层必须为字典：{path}")
    return value


def torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_state_flexible(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch_load(checkpoint_path, device)
    state = checkpoint.get(
        "model",
        checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)),
    )
    if not isinstance(state, dict):
        raise TypeError(f"检查点中找不到模型权重：{checkpoint_path}")
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    ignored = [
        key
        for key in unexpected
        if key.startswith("gesture_head.") or key.startswith("finger_head.")
    ]
    unexpected = [key for key in unexpected if key not in ignored]
    if missing or unexpected:
        raise RuntimeError(
            f"权重与模型不匹配：{checkpoint_path}\n"
            f"missing={missing}\nunexpected={unexpected}\n"
            f"ignored_old_heads={ignored}"
        )


def build_from_entry(
    key: str,
    entry: dict[str, Any],
    device: torch.device,
) -> LoadedModel:
    kind = str(entry.get("kind", "proposed")).lower()
    config_path = Path(entry["config"])
    checkpoint_path = Path(entry["checkpoint"])
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置：{config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到权重：{checkpoint_path}")

    cfg = load_config(config_path)
    if kind == "proposed":
        model = make_model(cfg)
    elif kind == "ablation1":
        model = build_ablation_model(cfg, str(entry.get("variant", key)))
    elif kind == "comparison":
        if build_repair_model is None:
            raise ImportError("项目中缺少 src/comparison_models.py")
        model = build_repair_model(str(entry.get("model_name", key)), cfg)
    else:
        raise ValueError(f"未知 kind={kind}：{key}")

    model = model.to(device)
    load_state_flexible(model, checkpoint_path, device)
    model.eval()
    return LoadedModel(
        key=key,
        label=str(entry.get("label", key)),
        model=model,
        checkpoint=checkpoint_path,
    )


def parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def choose_frames(
    total_frames: int,
    explicit: list[int],
    count: int,
) -> list[int]:
    if explicit:
        frames = [index for index in explicit if 0 <= index < total_frames]
        if not frames:
            raise ValueError("--frames 中没有合法帧编号")
        return list(dict.fromkeys(frames))
    count = max(1, min(int(count), total_frames))
    return np.linspace(0, total_frames - 1, count, dtype=int).tolist()


def output_tensor(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        for key in ("repaired", "pred_coords", "output"):
            if key in outputs:
                return outputs[key]
        raise KeyError(f"模型输出字典没有 repaired/pred_coords：{outputs.keys()}")
    if torch.is_tensor(outputs):
        return outputs
    raise TypeError(f"无法识别模型输出类型：{type(outputs)!r}")


def sequence_metrics(
    repaired: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
) -> tuple[float, float]:
    distance = torch.linalg.vector_norm(repaired - target, dim=-1)
    missing = visible_mask < 0.5
    if not missing.any():
        return 0.0, 0.0
    values = distance[missing]
    return float(values.mean()), float((values <= 0.1).float().mean())


def equal_limits(arrays: list[np.ndarray], padding: float = 0.0):
    pts = np.concatenate([np.asarray(x).reshape(-1, 3) for x in arrays], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]

    if not pts.size:
        return ((-1.0, 1.0),) * 3

    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-3)
    radius *= 1.0 + padding

    return tuple(
        (center[i] - radius, center[i] + radius)
        for i in range(3)
    )


def plot_skeleton(
    ax,
    coords: np.ndarray,
    edges: list[list[int]],
    *,
    visible_mask: np.ndarray | None,
    title: str,
    limits,
    view_elev: float,
    view_azim: float,
    mark_missing: bool,
    draw_missing_edges: bool = False,
    ground_truth_coords: np.ndarray | None = None,
    show_ground_truth_missing: bool = False,
    show_error_lines: bool = False,
    show_legend: bool = False,
    missing_marker_size: float = 80.0,
    missing_label: str = "Missing / repaired joint",
    missing_marker: str = "x",
    missing_color: str = MISSING_COLOR,
    observed_color: str = GROUND_TRUTH_COLOR,
    ground_truth_color: str = GROUND_TRUTH_COLOR,
    bone_color: str = BONE_COLOR,
    error_color: str = ERROR_COLOR,
    show_grid: bool = True,
    grid_ticks: int = 5,
) -> None:
    """绘制单帧三维骨架，并显式标注缺失/修复关节。

    标记规则：
    - 绿色圆点：真实/原始可见关节；
    - 红色 ×：Masked 面板中的缺失关节；
    - 蓝色 △：模型输出中的修补关节；
    - 绿色空心圆：缺失关节的真实位置；
    - 紫色虚线：修补位置到真实位置的误差。
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords 应为 [V, 3]，实际为 {coords.shape}")

    visible = (
        np.ones(coords.shape[0], dtype=bool)
        if visible_mask is None
        else np.asarray(visible_mask).astype(bool).reshape(-1)
    )
    if len(visible) != len(coords):
        raise ValueError(
            f"visible_mask 长度 {len(visible)} 与关节数 {len(coords)} 不一致"
        )
    missing = ~visible

    # Masked面板只画完全可见的骨边；模型输出面板可画修复后的完整骨架。
    for start, end in edges:
        start, end = int(start), int(end)
        if start >= len(coords) or end >= len(coords):
            continue
        if (
            visible_mask is not None
            and not draw_missing_edges
            and not (visible[start] and visible[end])
        ):
            continue
        ax.plot(
            [coords[start, 0], coords[end, 0]],
            [coords[start, 1], coords[end, 1]],
            [coords[start, 2], coords[end, 2]],
            color=bone_color,
            linewidth=3.0,
            alpha=0.85,
        )

    if visible.any():
        ax.scatter(
            coords[visible, 0],
            coords[visible, 1],
            coords[visible, 2],
            s=45,
            marker="o",
            color=observed_color,
            edgecolors="none",
            depthshade=False,
            label="Observed joint" if visible_mask is not None else "Ground-truth joint",
        )

    if mark_missing and missing.any():
        ax.scatter(
            coords[missing, 0],
            coords[missing, 1],
            coords[missing, 2],
            s=float(missing_marker_size),
            marker=missing_marker,
            color=missing_color,
            linewidths=1.8,
            depthshade=False,
            label=missing_label,
        )

    gt = None
    if ground_truth_coords is not None:
        gt = np.asarray(ground_truth_coords, dtype=np.float32)
        if gt.shape != coords.shape:
            raise ValueError(
                f"ground_truth_coords 形状 {gt.shape} 与 coords {coords.shape} 不一致"
            )

    if gt is not None and show_ground_truth_missing and missing.any():
        ax.scatter(
            gt[missing, 0],
            gt[missing, 1],
            gt[missing, 2],
            s=float(missing_marker_size) * 0.95,
            marker="o",
            facecolors="none",
            edgecolors=ground_truth_color,
            linewidths=1.6,
            depthshade=False,
            label="Ground-truth position",
        )

    if gt is not None and show_error_lines and missing.any():
        first_error_line = True
        for joint_index in np.flatnonzero(missing):
            predicted_point = coords[joint_index]
            target_point = gt[joint_index]
            if not (
                np.isfinite(predicted_point).all()
                and np.isfinite(target_point).all()
            ):
                continue
            ax.plot(
                [predicted_point[0], target_point[0]],
                [predicted_point[1], target_point[1]],
                [predicted_point[2], target_point[2]],
                color=error_color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label="Prediction error" if first_error_line else None,
            )
            first_error_line = False

    limits_array = np.asarray(limits, dtype=float)
    if limits_array.shape != (3, 2):
        raise ValueError(
            f"limits 必须为 ((xmin, xmax), (ymin, ymax), (zmin, zmax))，"
            f"实际形状为 {limits_array.shape}"
        )

    ax.set_xlim(limits_array[0, 0], limits_array[0, 1])
    ax.set_ylim(limits_array[1, 0], limits_array[1, 1])
    ax.set_zlim(limits_array[2, 0], limits_array[2, 1])
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_title(title, fontsize=24,pad=8,fontweight="normal")

    # 保留刻度位置才能显示 Matplotlib 3D 网格。
    # 这里只隐藏刻度数字，不删除刻度本身。
    if show_grid:
        tick_count = max(2, int(grid_ticks))
        ax.set_xticks(np.linspace(limits_array[0, 0], limits_array[0, 1], tick_count))
        ax.set_yticks(np.linspace(limits_array[1, 0], limits_array[1, 1], tick_count))
        ax.set_zticks(np.linspace(limits_array[2, 0], limits_array[2, 1], tick_count))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.tick_params(axis="x", which="both", length=0, pad=-2)
        ax.tick_params(axis="y", which="both", length=0, pad=-2)
        ax.tick_params(axis="z", which="both", length=0, pad=-2)
        ax.grid(True)

        # 显示立体坐标盒与浅色格子面。
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_alpha(0.10)
            axis._axinfo["grid"]["linestyle"] = "--"
            axis._axinfo["grid"]["linewidth"] = 0.65
            axis._axinfo["grid"]["color"] = (0.35, 0.35, 0.35, 0.50)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)

    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.35)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            if label and label not in unique:
                unique[label] = handle
        if unique:
            ax.legend(
                unique.values(),
                unique.keys(),
                loc="upper right",
                fontsize=24,
                frameon=False,
            )


def pages(items: list[Any], page_size: int):
    for start in range(0, len(items), page_size):
        yield start // page_size + 1, items[start : start + page_size]


def save_panel_pages(
    *,
    target: np.ndarray,
    masked: np.ndarray,
    visible_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    model_labels: dict[str, str],
    model_metrics: dict[str, tuple[float, float]],
    edges: list[list[int]],
    frame_ids: list[int],
    rows: int,
    cols: int,
    dpi: int,
    output_stem: Path,
    view_elev: float,
    view_azim: float,
    mark_missing_points: bool,
    show_ground_truth_missing: bool,
    show_error_lines: bool,
    show_legend: bool,
    missing_marker_size: float,
    show_grid: bool,
    grid_ticks: int,
) -> list[Path]:
    sources = ["Ground truth", "Masked", *predictions.keys()]
    panels: list[tuple[str, int]] = [
        (source, frame_id) for frame_id in frame_ids for source in sources
    ]

    if rows <= 0 and cols <= 0:
        rows, cols = len(frame_ids), len(sources)
    elif rows <= 0:
        rows = math.ceil(len(panels) / max(cols, 1))
    elif cols <= 0:
        cols = math.ceil(len(panels) / max(rows, 1))
    rows, cols = max(1, rows), max(1, cols)
    page_size = rows * cols

    # 每一帧只根据该帧的真实骨架确定坐标范围。
    # 同一帧的 GT、Masked 和所有模型面板共用该范围，便于公平比较。
    frame_limits = {
        frame_id: equal_limits([target[frame_id]], padding=0.0)
        for frame_id in frame_ids
    }
    saved: list[Path] = []
    for page_index, page_items in pages(panels, page_size):
        figure = plt.figure(figsize=(6.0 * cols, 5.5 * rows))
        for panel_index, (source, frame_id) in enumerate(page_items, start=1):
            ax = figure.add_subplot(rows, cols, panel_index, projection="3d")
            if source == "Ground truth":
                coords = target[frame_id]
                mask = None
                title = f"GT | frame {frame_id}"
                panel_mark_missing = False
                draw_missing_edges = True
                gt_coords = None
                panel_show_gt = False
                panel_show_error = False
                missing_label = "Missing joint"
                panel_missing_marker = "x"
                panel_missing_color = MISSING_COLOR
            elif source == "Masked":
                # 使用完整真实坐标定位缺失点，但模型输入仍然是 masked。
                # 可见关节正常显示，原本缺失的关节用 × 标注；
                # 与缺失关节相连的骨边不绘制。
                coords = target[frame_id]
                mask = visible_mask[frame_id]
                title = f"Masked | frame {frame_id}"
                panel_mark_missing = mark_missing_points
                draw_missing_edges = False
                gt_coords = None
                panel_show_gt = False
                panel_show_error = False
                missing_label = "Missing joint"
                panel_missing_marker = "x"
                panel_missing_color = MISSING_COLOR
            else:
                # 模型输出面板：
                # 绿色圆点=原始可见关节，蓝色△=模型修补关节，
                # 空心圆=真实位置，虚线=预测误差。
                coords = predictions[source][frame_id]
                mask = visible_mask[frame_id]
                mpjpe, pck01 = model_metrics[source]
                # title = (
                #     f"{model_labels[source]} | frame {frame_id}\n"
                #     f"MPJPE={mpjpe:.4f}, PCK@0.1={pck01:.4f}"
                # )
                title = (
                    f"{model_labels[source]}"
                    )
                panel_mark_missing = mark_missing_points
                draw_missing_edges = True
                gt_coords = target[frame_id]
                panel_show_gt = show_ground_truth_missing
                panel_show_error = show_error_lines
                missing_label = "Repaired joint"
                panel_missing_marker = "^"
                panel_missing_color = REPAIRED_COLOR
            panel_limits = frame_limits[frame_id]

            plot_skeleton(
                ax,
                coords,
                edges,
                visible_mask=mask,
                title=title,
                limits=panel_limits,
                view_elev=view_elev,
                view_azim=view_azim,
                mark_missing=panel_mark_missing,
                draw_missing_edges=draw_missing_edges,
                ground_truth_coords=gt_coords,
                show_ground_truth_missing=panel_show_gt,
                show_error_lines=panel_show_error,
                show_legend=show_legend and source != "Ground truth",
                missing_marker_size=missing_marker_size,
                missing_label=missing_label,
                missing_marker=panel_missing_marker,
                missing_color=panel_missing_color,
                observed_color=GROUND_TRUTH_COLOR,
                ground_truth_color=GROUND_TRUTH_COLOR,
                bone_color=BONE_COLOR,
                error_color=ERROR_COLOR,
                show_grid=show_grid,
                grid_ticks=grid_ticks,
            )

        for empty_index in range(len(page_items) + 1, page_size + 1):
            ax = figure.add_subplot(rows, cols, empty_index)
            ax.axis("off")
        figure.tight_layout()
        output = output_stem.with_name(
            f"{output_stem.name}_page_{page_index:02d}.png"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        saved.append(output)
    return saved


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载提出模型、比较模型或消融模型，生成可调行列的三维骨架输出图"
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", default="proposed", help="逗号分隔 manifest 键名")
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sample-indices", default="", help="如 0,10,100")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--mask-type", default="whole_finger")
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--mask-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--frames", default="0,4,8,12,16,20,24,28,31", help="如 0,4,8,12,16,20,24,28,31")
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--rows", type=int, default=0, help="0 表示自动")
    parser.add_argument("--cols", type=int, default=0, help="0 表示自动")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--view-elev", type=float, default=18.0)
    parser.add_argument("--view-azim", type=float, default=-65.0)
    parser.add_argument(
        "--mark-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否用红色×标注缺失关节、用蓝色△标注修补关节",
    )
    parser.add_argument(
        "--show-gt-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在模型输出中用空心圆显示缺失关节真实位置",
    )
    parser.add_argument(
        "--show-error-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否绘制修复位置到真实位置的虚线误差",
    )
    parser.add_argument(
        "--show-legend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否在每个Masked/模型面板显示图例",
    )
    parser.add_argument(
        "--missing-marker-size",
        type=float,
        default=80.0,
        help="缺失点、修复点和真实位置标记大小",
    )
    parser.add_argument(
        "--show-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示三维格子网格；使用 --no-show-grid 可关闭",
    )
    parser.add_argument(
        "--grid-ticks",
        type=int,
        default=5,
        help="每个坐标轴的网格刻度数，建议 4～7",
    )
    parser.add_argument("--save-npz", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_yaml(args.manifest)
    entries = manifest.get("models", manifest)
    if not isinstance(entries, dict) or not entries:
        raise ValueError("manifest 中必须包含非空 models 字典")

    selected_keys = (
        list(entries.keys())
        if args.models.strip().lower() == "all"
        else [item.strip() for item in args.models.split(",") if item.strip()]
    )
    missing_keys = [key for key in selected_keys if key not in entries]
    if missing_keys:
        raise KeyError(f"manifest 中不存在模型：{missing_keys}")

    data_config = args.data_config or manifest.get("data_config")
    if not data_config:
        first_entry = entries[selected_keys[0]]
        data_config = first_entry["config"]
    data_cfg = load_config(data_config)
    if args.data_root:
        data_cfg["data"]["root"] = args.data_root

    set_seed(args.seed)
    device = resolve_device(data_cfg.get("device", "auto"))
    print("设备：", device)
    loaded_models = {
        key: build_from_entry(key, entries[key], device) for key in selected_keys
    }
    dataset = make_dataset(data_cfg, args.split)
    mask_generator = make_mask_generator(data_cfg)
    edges = data_cfg["data"]["edges"]

    explicit_samples = parse_int_list(args.sample_indices)
    if explicit_samples:
        sample_indices = explicit_samples
    else:
        sample_indices = list(
            range(args.sample_start, args.sample_start + args.num_samples)
        )
    sample_indices = [index for index in sample_indices if 0 <= index < len(dataset)]
    if not sample_indices:
        raise IndexError("没有合法样本编号")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []

    for sample_index in sample_indices:
        sample = dataset[sample_index]
        skeleton = sample["skeleton"].unsqueeze(0).to(device)
        total_frames = int(skeleton.shape[1])
        frame_ids = choose_frames(
            total_frames,
            parse_int_list(args.frames),
            args.num_frames,
        )

        for repeat in range(max(1, args.mask_repeats)):
            mask_seed = args.seed + sample_index * 1000 + repeat
            visible_mask, chosen_types, chosen_ratios = mask_generator.generate(
                batch_size=1,
                num_frames=skeleton.shape[1],
                num_joints=skeleton.shape[2],
                device=device,
                seed=mask_seed,
                force_type=args.mask_type,
                force_ratio=args.mask_ratio,
            )
            masked = skeleton * visible_mask.unsqueeze(-1)
            predictions: dict[str, np.ndarray] = {}
            model_metrics: dict[str, tuple[float, float]] = {}
            model_labels: dict[str, str] = {}

            with torch.inference_mode():
                for key, loaded in loaded_models.items():
                    repaired = output_tensor(loaded.model(masked, visible_mask))
                    mpjpe, pck01 = sequence_metrics(
                        repaired, skeleton, visible_mask
                    )
                    predictions[key] = repaired[0].detach().cpu().numpy()
                    model_metrics[key] = (mpjpe, pck01)
                    model_labels[key] = loaded.label
                    metric_rows.append(
                        {
                            "sample_index": sample_index,
                            "repeat": repeat,
                            "mask_seed": mask_seed,
                            "mask_type": chosen_types[0],
                            "mask_ratio": float(chosen_ratios[0]),
                            "model": key,
                            "label": loaded.label,
                            "mpjpe": mpjpe,
                            "pck@0.1": pck01,
                            "checkpoint": str(loaded.checkpoint),
                        }
                    )

            target_np = skeleton[0].detach().cpu().numpy()
            masked_np = masked[0].detach().cpu().numpy()
            mask_np = visible_mask[0].detach().cpu().numpy()
            stem = output_dir / f"sample_{sample_index:04d}_repeat_{repeat:02d}"
            saved = save_panel_pages(
                target=target_np,
                masked=masked_np,
                visible_mask=mask_np,
                predictions=predictions,
                model_labels=model_labels,
                model_metrics=model_metrics,
                edges=edges,
                frame_ids=frame_ids,
                rows=args.rows,
                cols=args.cols,
                dpi=args.dpi,
                output_stem=stem,
                view_elev=args.view_elev,
                view_azim=args.view_azim,
                mark_missing_points=args.mark_missing,
                show_ground_truth_missing=args.show_gt_missing,
                show_error_lines=args.show_error_lines,
                show_legend=args.show_legend,
                missing_marker_size=args.missing_marker_size,
                show_grid=args.show_grid,
                grid_ticks=args.grid_ticks,
            )
            for path in saved:
                print("已保存：", path.resolve())

            if args.save_npz:
                np.savez_compressed(
                    stem.with_suffix(".npz"),
                    ground_truth=target_np,
                    masked=masked_np,
                    visible_mask=mask_np,
                    frame_ids=np.asarray(frame_ids),
                    **{f"prediction__{key}": value for key, value in predictions.items()},
                )

    write_metrics(output_dir / "visualization_metrics.csv", metric_rows)
    print("样本指标：", (output_dir / "visualization_metrics.csv").resolve())


if __name__ == "__main__":
    main()

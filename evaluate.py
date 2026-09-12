from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.factory import make_dataset, make_mask_generator, make_model
from src.metrics import (
    # classification_accuracy,
    missing_mpjpe,
    missing_velocity_error,
)
from src.utils import load_checkpoint, resolve_device, save_json


@torch.no_grad()
def evaluate_setting(
    model,
    loader,
    generator,
    device,
    mask_type: str,
    ratio: float,
    seed: int,
) -> dict[str, float]:
    """在指定缺失类型和缺失比例下评估整个测试集。"""
    model.eval()

    sums = {
        "mpjpe": 0.0,
        "velocity_error": 0.0,
        # "gesture_acc": 0.0,
        # "finger_acc": 0.0,
    }
    sample_count = 0

    for batch_index, batch in enumerate(loader):
        x = batch["skeleton"].to(device)
        # gesture = batch["gesture"].to(device)
        # finger = batch["finger"].to(device)

        mask, _, _ = generator.generate(
            batch_size=x.shape[0],
            num_frames=x.shape[1],
            num_joints=x.shape[2],
            device=device,
            seed=seed + batch_index,
            force_type=mask_type,
            force_ratio=ratio,
        )
        
        masked = x * mask.unsqueeze(-1)
        outputs = model(masked, mask)

        batch_size = x.shape[0]
        sums["mpjpe"] += float(
            missing_mpjpe(outputs["repaired"], x, mask)
        ) * batch_size
        sums["velocity_error"] += float(
            missing_velocity_error(outputs["repaired"], x, mask)
        ) * batch_size
        # sums["gesture_acc"] += float(
        #     classification_accuracy(outputs["gesture_logits"], gesture)
        # ) * batch_size
        # sums["finger_acc"] += float(
        #     classification_accuracy(outputs["finger_logits"], finger)
        # ) * batch_size

        sample_count += batch_size

    denominator = max(sample_count, 1)
    return {
        key: value / denominator
        for key, value in sums.items()
    }


def parse_frame_ids(
    value: str,
    total_frames: int,
) -> list[int]:
    """
    把 "0,4,8,12" 转换为帧编号列表。
    超出范围的帧会自动忽略。
    """
    frame_ids: list[int] = []

    for item in value.split(","):
        item = item.strip()
        if not item:
            continue

        try:
            frame_id = int(item)
        except ValueError as exc:
            raise ValueError(
                f"无法解析预览帧编号：{item}"
            ) from exc

        if 0 <= frame_id < total_frames:
            frame_ids.append(frame_id)

    # 去重并保持原顺序
    frame_ids = list(dict.fromkeys(frame_ids))

    if not frame_ids:
        frame_ids = np.linspace(
            0,
            total_frames - 1,
            min(8, total_frames),
            dtype=int,
        ).tolist()

    return frame_ids


def compute_frame_axis_limits(
    reference: np.ndarray,
    padding: float = 1.35,
) -> tuple[np.ndarray, float]:
    """
    根据当前帧的完整骨架确定显示范围。

    reference:
        [V, 3]，当前帧完整骨架。
    """
    finite = np.isfinite(reference).all(axis=1)
    points = reference[finite]

    if points.size == 0:
        return np.zeros(3, dtype=np.float32), 1.0

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0

    radius = float((maxs - mins).max()) / 2.0
    radius = max(radius * padding, 0.5)

    return center, radius

def draw_skeleton_panel(
    ax,
    points: np.ndarray,
    edges: list[list[int]],
    center: np.ndarray,
    radius: float,
    title: str,
    visible: np.ndarray | None = None,
) -> None:
    """
    绘制一帧骨架。

    visible:
        [V]，True表示该关节点可见。
        缺失输入图中会隐藏缺失关节点及相关骨骼边。
    """
    num_joints = points.shape[0]

    if visible is None:
        visible = np.ones(num_joints, dtype=bool)
    else:
        visible = visible.astype(bool)

    finite = np.isfinite(points).all(axis=1)
    valid = visible & finite

    if valid.any():
        ax.scatter(
            points[valid, 0],
            points[valid, 1],
            points[valid, 2],
            s=32,
        )

    for i, j in edges:
        i = int(i)
        j = int(j)

        if not (0 <= i < num_joints and 0 <= j < num_joints):
            continue
        if not (valid[i] and valid[j]):
            continue

        segment = points[[i, j]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            linewidth=1.8,
        )

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.tick_params(labelsize=7)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=20, azim=-60)


def save_multiframe_repair_preview(
    complete: np.ndarray,
    masked: np.ndarray,
    repaired: np.ndarray,
    visible_mask: np.ndarray,
    edges: list[list[int]],
    frame_ids: list[int],
    output_path: str | Path,
    sample_title: str = "",
) -> None:
    """
    保存若干帧的三列对比图：

    Complete | Masked | Repaired
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

   
    rows = len(frame_ids)
    fig = plt.figure(
        figsize=(15, 4.2 * rows),
    )

    for row, frame_id in enumerate(frame_ids):
    # 每一帧按照完整骨架确定显示范围
    # 同一行三列仍共用同一个范围，方便公平比较
        center, radius = compute_frame_axis_limits(
            complete[frame_id]
        )

        arrays = [
            complete[frame_id],
            masked[frame_id],
            repaired[frame_id],
        ]
        visibility = [
            None,
            visible_mask[frame_id] > 0.5,
            None,
        ]
        titles = [
            f"Complete | Frame {frame_id}",
            f"Masked | Frame {frame_id}",
            f"Repaired | Frame {frame_id}",
        ]

        for col in range(3):
            ax = fig.add_subplot(
                rows,
                3,
                row * 3 + col + 1,
                projection="3d",
            )

            draw_skeleton_panel(
                ax=ax,
                points=arrays[col],
                edges=edges,
                center=center,
                radius=radius,
                title=titles[col],
                visible=visibility[col],
            )

    if sample_title:
        fig.suptitle(
            sample_title,
            fontsize=14,
        )
        fig.tight_layout(
            rect=[0, 0, 1, 0.98],
        )
    else:
        fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


@torch.no_grad()
def create_preview(
    model,
    dataset,
    generator,
    device,
    edges: list[list[int]],
    sample_index: int,
    mask_type: str,
    ratio: float,
    seed: int,
    frame_text: str,
    output_path: str | Path,
) -> None:
    """选取一个测试样本，生成多帧缺失与修补对比图。"""
    if not 0 <= sample_index < len(dataset):
        raise IndexError(
            f"preview-index={sample_index} 超出范围，"
            f"测试集共有 {len(dataset)} 个样本。"
        )

    model.eval()

    sample = dataset[sample_index]
    x = sample["skeleton"].unsqueeze(0).to(device)

    mask, chosen_types, chosen_ratios = generator.generate(
        batch_size=1,
        num_frames=x.shape[1],
        num_joints=x.shape[2],
        device=device,
        seed=seed,
        force_type=mask_type,
        force_ratio=ratio,
    )

    masked = x * mask.unsqueeze(-1)
    outputs = model(masked, mask)
    repaired = outputs["repaired"]

    complete_np = x[0].detach().cpu().numpy()
    masked_np = masked[0].detach().cpu().numpy()
    repaired_np = repaired[0].detach().cpu().numpy()
    mask_np = mask[0].detach().cpu().numpy()

    print(
        "完整骨架范围：",
        complete_np.min(axis=(0, 1)),
        complete_np.max(axis=(0, 1)),
    )

    print(
        "修补骨架范围：",
        repaired_np.min(axis=(0, 1)),
        repaired_np.max(axis=(0, 1)),
    )

    print(
        "缺失关节点数量：",
        int((mask_np < 0.5).sum()),
    )

    print(
        "修补坐标最大绝对值：",
        float(np.abs(repaired_np).max()),
    )



    frame_ids = parse_frame_ids(
        frame_text,
        total_frames=complete_np.shape[0],
    )

    gesture_id = int(sample["gesture"]) + 1
    finger_id = int(sample["finger"]) + 1
    subject_id = int(sample["subject"])
    trial_id = int(sample["trial"])

    sample_mpjpe = float(
        missing_mpjpe(
            repaired,
            x,
            mask,
        )
    )

    title = (
        f"Gesture={gesture_id}, Finger={finger_id}, "
        f"Subject={subject_id}, Trial={trial_id}, "
        f"Mask={chosen_types[0]}, Ratio={chosen_ratios[0]:.2f}, "
        f"MPJPE={sample_mpjpe:.6f}"
    )

    save_multiframe_repair_preview(
        complete=complete_np,
        masked=masked_np,
        repaired=repaired_np,
        visible_mask=mask_np,
        edges=edges,
        frame_ids=frame_ids,
        output_path=output_path,
        sample_title=title,
    )

    print("多帧修补预览已保存：", Path(output_path).resolve())
    print("预览帧：", frame_ids)
    print("该样本缺失点 MPJPE：", sample_mpjpe)
    print("样本文件：", sample["path"])


def check_paths(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    config = Path(args.config)
    data_root = Path(args.data_root)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"找不到模型检查点：{checkpoint.resolve()}\n"
            "请先训练模型，或通过 --checkpoint 指定正确路径。"
        )

    if not config.exists():
        raise FileNotFoundError(
            f"找不到配置文件：{config.resolve()}"
        )

    if not data_root.exists():
        raise FileNotFoundError(
            f"找不到数据集目录：{data_root.resolve()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估DHG2016骨架修补模型并生成多帧修补预览"
    )

    # 设置默认值后，可以直接执行：python evaluate.py
    parser.add_argument(
        "--checkpoint",
        default="./runs/dhg2016_repairformer/best.pt",
    )
    parser.add_argument(
        "--config",
        default="./configs/default.yaml",
    )
    parser.add_argument(
        "--data-root",
        default="./DHG2016",
    )
    parser.add_argument(
        "--output",
        default="./runs/experiment_01/evaluation_results.json",
    )

    # 多帧预览参数
    parser.add_argument(
        "--preview-output",
        default="./runs/experiment_01/evaluation_preview.png",
    )
    parser.add_argument(
        "--preview-index",
        type=int,
        default=0,
        help="测试集中的样本序号",
    )
    parser.add_argument(
        "--preview-mask-type",
        default="whole_finger",
        choices=[
            "random_joint",
            "whole_finger",
            "temporal_block",
            "fingertips",
            "whole_frame",
        ],
    )
    parser.add_argument(
        "--preview-ratio",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--preview-frames",
        default="0,4,8,12,16,20,24,31",
        help="用逗号分隔的帧编号",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="只评估，不生成多帧预览图",
    )

    args = parser.parse_args()
    check_paths(args)

    cfg = load_config(args.config)
    cfg["data"]["root"] = args.data_root

    device = resolve_device(
        cfg.get("device", "auto")
    )
    print("使用设备：", device)

    dataset = make_dataset(
        cfg,
        "test",
    )
    print("测试集：", dataset.summary())

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    model = make_model(cfg).to(device)
    load_checkpoint(
        args.checkpoint,
        model,
        map_location=device,
    )

    generator = make_mask_generator(cfg)

    results: dict[str, dict[str, dict[str, float]]] = {}

    for mask_type in cfg["eval"]["mask_types"]:
        results[mask_type] = {}

        for ratio in cfg["eval"]["ratios"]:
            ratio = float(ratio)

            metrics = evaluate_setting(
                model=model,
                loader=loader,
                generator=generator,
                device=device,
                mask_type=mask_type,
                ratio=ratio,
                seed=cfg["eval"]["fixed_seed"],
            )

            results[mask_type][str(ratio)] = metrics
            print(mask_type, ratio, metrics)
    # base_seed = cfg["eval"]["fixed_seed"]
    # for mask_idx, mask_type in enumerate(cfg["eval"]["mask_types"]):
    #     results[mask_type] = {}
    #     for ratio_idx, ratio in enumerate(cfg["eval"]["ratios"]):
    #         ratio = float(ratio)
    #         # 组合唯一种子：mask类型索引 + 比例索引，确保每组参数seed不同
    #         unique_seed = base_seed + mask_idx * 1000 + ratio_idx
    #         metrics = evaluate_setting(
    #             model=model,
    #             loader=loader,
    #             generator=generator,
    #             device=device,
    #             mask_type=mask_type,
    #             ratio=ratio,
    #             seed=unique_seed,  # 使用独立种子
    #         )
    #         results[mask_type][str(ratio)] = metrics
    #         print(mask_type, ratio, metrics)


    save_json(
        Path(args.output),
        results,
    )
    print("评估结果已保存：", Path(args.output).resolve())

    if not args.skip_preview:
        create_preview(
            model=model,
            dataset=dataset,
            generator=generator,
            device=device,
            edges=cfg["data"]["edges"],
            sample_index=args.preview_index,
            mask_type=args.preview_mask_type,
            ratio=args.preview_ratio,
            seed=cfg["eval"]["fixed_seed"],
            frame_text=args.preview_frames,
            output_path=args.preview_output,
        )


if __name__ == "__main__":
    main()

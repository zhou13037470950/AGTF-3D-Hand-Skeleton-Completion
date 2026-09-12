from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter


# 解决 Windows 下 Matplotlib 中文显示问题
font_path = r"C:\Windows\Fonts\msyh.ttc"

if Path(font_path).exists():
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name

plt.rcParams["axes.unicode_minus"] = False

# ========== 仅新增：整体字体放大 ==========
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.tick_params(labelsize=12)

def draw_skeleton(ax, points: np.ndarray, edges: list[list[int]], title: str = "") -> None:
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=18)
    for i, j in edges:
        segment = points[[i, j]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2])
    _set_equal_axes(ax, points)
    ax.set_title(title)


def save_repair_comparison(
    complete: np.ndarray,
    masked: np.ndarray,
    repaired: np.ndarray,
    visible_mask: np.ndarray,
    edges: list[list[int]],
    path: str | Path,
    frame_ids: list[int] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = complete.shape[0]
    # frame_ids = frame_ids or [0, t // 3, 2 * t // 3, t - 1]
    frame_ids = frame_ids or [16, 20, 24,28,31]
    fig = plt.figure(figsize=(15, 4 * len(frame_ids)))
    for row, frame in enumerate(frame_ids):
        arrays = [complete[frame], masked[frame], repaired[frame]]
        titles = [f"Complete skeleton frame={frame}", "Missing input", "Repair results"]
        for col, (array, title) in enumerate(zip(arrays, titles)):
            ax = fig.add_subplot(len(frame_ids), 3, row * 3 + col + 1, projection="3d")
            draw_skeleton(ax, array, edges, title)
            if col == 1:
                missing_ids = np.where(visible_mask[frame] < 0.5)[0]
                if len(missing_ids):
                    ax.scatter(
                        complete[frame, missing_ids, 0],
                        complete[frame, missing_ids, 1],
                        complete[frame, missing_ids, 2],
                        marker="x",
                        s=35,
                    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_repair_animation(
    complete: np.ndarray,
    masked: np.ndarray,
    repaired: np.ndarray,
    edges: list[list[int]],
    path: str | Path,
    fps: int = 8,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 4))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    titles = ["Complete skeleton", "Missing input", "Repair results"]
    all_points = np.concatenate([complete, repaired], axis=0)

    def update(frame: int):
        for ax, data, title in zip(axes, [complete, masked, repaired], titles):
            ax.clear()
            draw_skeleton(ax, data[frame], edges, f"{title} frame={frame}")
            _set_equal_axes(ax, all_points.reshape(-1, 3))
        return axes

    animation = FuncAnimation(fig, update, frames=complete.shape[0], interval=1000 / fps)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)

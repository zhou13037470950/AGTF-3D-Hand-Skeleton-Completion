from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleInfo:
    path: Path
    gesture: int
    finger: int
    subject: int
    trial: int


_PATTERNS = {
    "gesture": re.compile(r"(?:gesture|gest)[_\- ]?(\d+)", re.I),
    "finger": re.compile(r"(?:finger|config)[_\- ]?(\d+)", re.I),
    "subject": re.compile(r"(?:subject|subj|user)[_\- ]?(\d+)", re.I),
    "trial": re.compile(r"(?:essai|trial|take)[_\- ]?(\d+)", re.I),
}


def _extract_metadata(path: Path) -> tuple[int, int, int, int]:
    text_parts = list(path.parts)
    found: dict[str, int] = {}
    for part in text_parts:
        for key, pattern in _PATTERNS.items():
            match = pattern.search(part)
            if match:
                found[key] = int(match.group(1))
    missing = [k for k in ("gesture", "finger", "subject", "trial") if k not in found]
    if missing:
        raise ValueError(
            f"无法从路径解析 {missing}: {path}\n"
            "期望目录中包含 gesture_1/finger_1/subject_1/essai_1 等字段。"
        )
    return found["gesture"], found["finger"], found["subject"], found["trial"]

def random_rotation(
    x,
    angle_range=30
):
    """
    x:
    T,J,3
    """

    angles=np.deg2rad(
        np.random.uniform(
            -angle_range,
            angle_range,
            size=3
        )
    )

    cx,cy,cz=np.cos(angles)
    sx,sy,sz=np.sin(angles)


    Rx=np.array([
        [1,0,0],
        [0,cx,-sx],
        [0,sx,cx]
    ], dtype=np.float32)

    Ry=np.array([
        [cy,0,sy],
        [0,1,0],
        [-sy,0,cy]
    ], dtype=np.float32)

    Rz=np.array([
        [cz,-sz,0],
        [sz,cz,0],
        [0,0,1]
    ], dtype=np.float32)

    R=Rz@Ry@Rx

    return np.einsum(
        "tjc,cd->tjd",
        x,
        R
    )
def random_scale(
    x,
    low=0.95,
    high=1.05
):

    scale=np.random.uniform(
        low,
        high
    )

    return x*scale

def add_noise(
    x,
    sigma=0.005
):

    noise=np.random.normal(
        0,
        sigma,
        x.shape
    )

    return x+noise.astype(
        np.float32
    )
def temporal_speed_aug(
    x,
    speed_range=(0.8,1.2)
):
    """
    Temporal speed augmentation

    x:
    T,J,3
    """

    T = x.shape[0]

    speed = np.random.uniform(
        speed_range[0],
        speed_range[1]
    )

    new_length = max(
        2,
        int(T * speed)
    )

    x = temporal_resample(
        x,
        new_length
    )

    x = temporal_resample(
        x,
        T
    )

    return x.astype(np.float32)

def random_bone_scale(
    x,
    edges,
    strength=0.05
):
    """
    Randomly perturb bone lengths.
    
    x:
        T,J,3
    edges:
        skeleton connections
    """

    x = x.copy()

    for parent, child in edges:

        scale = np.random.uniform(
            1-strength,
            1+strength
        )

        bone = x[:, child] - x[:, parent]

        x[:, child] = (
            x[:, parent]
            +
            bone * scale
        )

    return x

def discover_samples(root: str | Path, skeleton_filename: str) -> list[SampleInfo]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {root}")
    files = sorted(root.rglob(skeleton_filename))
    if not files:
        raise FileNotFoundError(
            f"在 {root} 下没有找到 {skeleton_filename}。请检查 data.root 和文件名。"
        )
    samples: list[SampleInfo] = []
    errors: list[str] = []
    for file in files:
        try:
            g, f, s, e = _extract_metadata(file)
            samples.append(SampleInfo(file, g, f, s, e))
        except ValueError as exc:
            errors.append(str(exc))
    if not samples:
        raise ValueError("找到了骨架文件，但没有任何路径能解析出标签。\n" + "\n".join(errors[:5]))
    if errors:
        print(f"警告：跳过 {len(errors)} 个无法解析标签的文件。")
    return samples


def load_skeleton_world(path: str | Path, num_joints: int = 22) -> np.ndarray:
    """兼容常见的 T×66 或 (T*22)×3 两种格式。"""
    path = Path(path)
    try:
        array = np.loadtxt(path, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"读取骨架文件失败: {path}: {exc}") from exc

    if array.ndim == 1:
        if array.size % (num_joints * 3) != 0:
            raise ValueError(f"文件元素数无法被 {num_joints * 3} 整除: {path}")
        array = array.reshape(-1, num_joints * 3)

    if array.ndim != 2:
        raise ValueError(f"骨架文件应是二维数值矩阵，实际形状 {array.shape}: {path}")

    if array.shape[1] == num_joints * 3:
        skeleton = array.reshape(array.shape[0], num_joints, 3)
    elif array.shape[1] == 3 and array.shape[0] % num_joints == 0:
        skeleton = array.reshape(-1, num_joints, 3)
    elif array.shape[0] == num_joints and array.shape[1] % 3 == 0:
        skeleton = array.T.reshape(-1, num_joints, 3)
    else:
        raise ValueError(
            f"无法识别骨架格式 {array.shape}: {path}。"
            f"期望 T×{num_joints * 3} 或 (T*{num_joints})×3。"
        )

    if not np.isfinite(skeleton).all():
        raise ValueError(f"骨架中存在 NaN/Inf: {path}")
    return skeleton.astype(np.float32, copy=False)


def temporal_resample(sequence: np.ndarray, target_length: int) -> np.ndarray:
    source_length = sequence.shape[0]
    if source_length == target_length:
        return sequence.astype(np.float32, copy=True)
    if source_length < 2:
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)

    source_t = np.linspace(0.0, 1.0, source_length, dtype=np.float32)
    target_t = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    flattened = sequence.reshape(source_length, -1)
    result = np.empty((target_length, flattened.shape[1]), dtype=np.float32)
    for index in range(flattened.shape[1]):
        result[:, index] = np.interp(target_t, source_t, flattened[:, index])
    return result.reshape(target_length, sequence.shape[1], sequence.shape[2])


def normalize_sequence(
    sequence: np.ndarray,
    root_joint: int,
    edges: Iterable[Iterable[int]],
    center_mode: str = "per_frame_root",
    scale_mode: str = "median_bone",
) -> np.ndarray:
    x = sequence.astype(np.float32, copy=True)
    if center_mode == "first_root":
        x -= x[0:1, root_joint : root_joint + 1, :]
    elif center_mode == "per_frame_root":
        x -= x[:, root_joint : root_joint + 1, :]
    elif center_mode == "none":
        pass
    else:
        raise ValueError(f"未知 center_mode: {center_mode}")

    if scale_mode == "median_bone":
        lengths = []
        for i, j in edges:
            lengths.append(np.linalg.norm(x[:, int(i)] - x[:, int(j)], axis=-1))
        if lengths:
            scale = float(np.median(np.concatenate(lengths)))
        else:
            scale = float(np.std(x))
    elif scale_mode == "std":
        scale = float(np.std(x))
    elif scale_mode == "none":
        scale = 1.0
    else:
        raise ValueError(f"未知 scale_mode: {scale_mode}")
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    x /= scale
    return x


class DHG2016Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        skeleton_filename: str = "skeleton_world.txt",
        sequence_length: int = 32,
        num_joints: int = 22,
        subjects: list[int] | None = None,
        root_joint: int = 0,
        edges: list[list[int]] | None = None,
        center_mode: str = "per_frame_root",
        scale_mode: str = "median_bone",
        preload: bool = True,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.sequence_length = int(sequence_length)
        self.num_joints = int(num_joints)
        self.root_joint = int(root_joint)
        self.edges = edges or []
        self.center_mode = center_mode
        self.scale_mode = scale_mode
        all_samples = discover_samples(self.root, skeleton_filename)
        subject_set = set(subjects) if subjects else None
        self.samples = [s for s in all_samples if subject_set is None or s.subject in subject_set]
        if not self.samples:
            raise ValueError(f"当前 subjects={subjects} 没有样本。")
        self.preload = bool(preload)
        self.augment = bool(augment)
        print(
    f"Dataset split augment = {self.augment}, samples={len(self.samples)}"
)
        self.cache: list[np.ndarray] | None = None
        if self.preload:
            self.cache = [self._load_and_process(sample) for sample in self.samples]

    def _load_and_process(self, sample: SampleInfo) -> np.ndarray:
        x = load_skeleton_world(sample.path, self.num_joints)
        x = temporal_resample(x, self.sequence_length)
        x = normalize_sequence(
            x,
            root_joint=self.root_joint,
            edges=self.edges,
            center_mode=self.center_mode,
            scale_mode=self.scale_mode,
        )
        return x

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        info = self.samples[index]
        # x = self.cache[index] if self.cache is not None else self._load_and_process(info)
        x = (
    self.cache[index].copy()
    if self.cache is not None
    else self._load_and_process(info)
)
        if self.augment:

             # temporal speed
            x = temporal_speed_aug(
                x,
                speed_range=(0.8,1.2)
            )
            # 1. rotation
            x=random_rotation(
                x,
                angle_range=30
            )


            # 2. global scale
            x=random_scale(
                x,
                0.95,
                1.05
            )


            # 3. bone length deformation
            x=random_bone_scale(
                x,
                self.edges,
                strength=0.05
            )


            # 4. sensor noise
            x=add_noise(
                x,
                sigma=0.003
            )

              
        return {
            "skeleton": torch.from_numpy(x.copy()),
            "gesture": torch.tensor(info.gesture - 1, dtype=torch.long),
            "finger": torch.tensor(info.finger - 1, dtype=torch.long),
            "subject": torch.tensor(info.subject, dtype=torch.long),
            "trial": torch.tensor(info.trial, dtype=torch.long),
            "path": str(info.path),
        }

    def summary(self) -> dict[str, object]:
        gestures = sorted({s.gesture for s in self.samples})
        fingers = sorted({s.finger for s in self.samples})
        subjects = sorted({s.subject for s in self.samples})
        return {
            "samples": len(self.samples),
            "gestures": gestures,
            "fingers": fingers,
            "subjects": subjects,
            "sequence_length": self.sequence_length,
            "num_joints": self.num_joints,
        }

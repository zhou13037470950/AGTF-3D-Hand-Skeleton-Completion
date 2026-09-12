from __future__ import annotations

from torch.utils.data import DataLoader

from .dataset import DHG2016Dataset
from .masks import MaskGenerator
from .model import RepairFormer


def make_dataset(cfg: dict, split: str) -> DHG2016Dataset:
    data = cfg["data"]
    subjects = None if split == "all" else data[f"{split}_subjects"]
    return DHG2016Dataset(
        root=data["root"],
        skeleton_filename=data["skeleton_filename"],
        sequence_length=data["sequence_length"],
        num_joints=data["num_joints"],
        subjects=subjects,
        root_joint=data["root_joint"],
        edges=data["edges"],
        center_mode=data["center_mode"],
        scale_mode=data["scale_mode"],
        preload=data.get("preload", True),
        augment=(split == "train"),
    )


def make_loader(cfg: dict, split: str, shuffle: bool) -> DataLoader:
    dataset = make_dataset(cfg, split)
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=True,
        drop_last=shuffle,
    )


def make_mask_generator(cfg: dict) -> MaskGenerator:
    return MaskGenerator(
        min_ratio=cfg["mask"]["min_ratio"],
        max_ratio=cfg["mask"]["max_ratio"],
        type_weights=cfg["mask"]["types"],
        finger_groups=cfg["data"]["finger_groups"],
        fingertips=cfg["data"]["fingertips"],
    )


def make_model(cfg: dict) -> RepairFormer:
    data, model = cfg["data"], cfg["model"]
    return RepairFormer(
        num_frames=data["sequence_length"],
        num_joints=data["num_joints"],
        edges=data["edges"],
        **model,
    )

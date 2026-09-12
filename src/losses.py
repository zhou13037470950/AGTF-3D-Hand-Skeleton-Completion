from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weights = mask.to(device=values.device, dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def joint_weights(
    num_joints: int,
    cfg: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = torch.ones(num_joints, device=device, dtype=dtype)
    weight = float(cfg.get("fingertip_weight", 1.0))
    indices = cfg.get("fingertip_indices", [5, 9, 13, 17, 21])
    valid = [int(index) for index in indices if 0 <= int(index) < num_joints]
    if valid:
        weights[valid] = weight
    return weights


def structural_losses(
    repaired: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    edges: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    missing = visible_mask < 0.5
    length_errors, direction_errors, valid_masks = [], [], []

    for i, j in edges:
        i, j = int(i), int(j)
        pred_bone = repaired[:, :, i] - repaired[:, :, j]
        true_bone = target[:, :, i] - target[:, :, j]
        pred_length = torch.linalg.vector_norm(pred_bone, dim=-1)
        true_length = torch.linalg.vector_norm(true_bone, dim=-1)

        length_errors.append((pred_length - true_length).abs())
        pred_direction = F.normalize(pred_bone, dim=-1, eps=1e-6)
        true_direction = F.normalize(true_bone, dim=-1, eps=1e-6)
        direction_errors.append(
            1.0 - (pred_direction * true_direction).sum(dim=-1)
        )
        valid_masks.append(missing[:, :, i] | missing[:, :, j])

    if not length_errors:
        zero = repaired.new_zeros(())
        return zero, zero

    valid = torch.stack(valid_masks, dim=-1)
    bone = masked_mean(torch.stack(length_errors, dim=-1), valid)
    direction = masked_mean(torch.stack(direction_errors, dim=-1), valid)
    return bone, direction

# losses.py
class BoneLengthLoss(nn.Module):
    def __init__(self, edges):
        super().__init__()
        self.edges = edges # 例如 YAML 中配置的 21 条骨骼边

    def forward(self, pred, target):
        # pred, target: (B, T, V, 3)
        pred_bones = pred[:, :, [e[1] for e in self.edges]] - pred[:, :, [e[0] for e in self.edges]]
        target_bones = target[:, :, [e[1] for e in self.edges]] - target[:, :, [e[0] for e in self.edges]]

        pred_lens = torch.norm(pred_bones, dim=-1)
        target_lens = torch.norm(target_bones, dim=-1)

        return F.l1_loss(pred_lens, target_lens)

# 总 Loss 组合：
# total_loss = l1_loss + 0.1 * velocity_loss + 0.2 * bone_length_loss


def compute_losses(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    edges: list[list[int]],
    loss_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_coords = outputs["pred_coords"]
    pred_velocity = outputs["pred_velocity"]
    repaired = outputs["repaired"]
    missing = 1.0 - visible_mask.float()

    weights = joint_weights(
        target.shape[2], loss_cfg, target.device, target.dtype
    ).view(1, 1, -1)
    coordinate_mask = missing * weights
    coordinate = masked_mean(
        F.smooth_l1_loss(pred_coords, target, reduction="none"),
        coordinate_mask,
    )

    target_velocity = target[:, 1:] - target[:, :-1]
    coordinate_velocity = pred_coords[:, 1:] - pred_coords[:, :-1]
    velocity_mask = 1.0 - visible_mask[:, 1:] * visible_mask[:, :-1]
    velocity = masked_mean(
        F.smooth_l1_loss(
            pred_velocity, target_velocity, reduction="none"
        ),
        velocity_mask,
    )
    consistency = masked_mean(
        F.smooth_l1_loss(
            pred_velocity, coordinate_velocity, reduction="none"
        ),
        velocity_mask,
    )

    pred_acceleration = (
        coordinate_velocity[:, 1:] - coordinate_velocity[:, :-1]
    )
    target_acceleration = (
        target_velocity[:, 1:] - target_velocity[:, :-1]
    )
    acceleration_mask = 1.0 - (
        visible_mask[:, 2:]
        * visible_mask[:, 1:-1]
        * visible_mask[:, :-2]
    )
    acceleration = masked_mean(
        F.smooth_l1_loss(
            pred_acceleration, target_acceleration, reduction="none"
        ),
        acceleration_mask,
    )

    bone, direction = structural_losses(
        repaired, target, visible_mask, edges
    )
    parts = {
        "coord": coordinate,
        "velocity": velocity,
        "acceleration": acceleration,
        "bone": bone,
        "direction": direction,
        "consistency": consistency,
    }
    total = sum(
        float(loss_cfg.get(name, 0.0)) * value
        for name, value in parts.items()
    )
    return total, parts

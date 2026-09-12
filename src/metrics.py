from __future__ import annotations

import torch


def missing_mpjpe(repaired: torch.Tensor, target: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
    missing = 1.0 - visible_mask
    distance = torch.linalg.vector_norm(repaired - target, dim=-1)
    return (distance * missing).sum() / missing.sum().clamp_min(1.0)


def missing_velocity_error(
    repaired: torch.Tensor, target: torch.Tensor, visible_mask: torch.Tensor
) -> torch.Tensor:
    pred_v = repaired[:, 1:] - repaired[:, :-1]
    true_v = target[:, 1:] - target[:, :-1]
    missing = 1.0 - visible_mask
    affected = torch.maximum(missing[:, 1:], missing[:, :-1])
    distance = torch.linalg.vector_norm(pred_v - true_v, dim=-1)
    return (distance * affected).sum() / affected.sum().clamp_min(1.0)


def classification_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == target).float().mean()

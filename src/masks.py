from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


@dataclass
class MaskGenerator:
    min_ratio: float
    max_ratio: float
    type_weights: dict[str, float]
    finger_groups: list[list[int]]
    fingertips: list[int]

    def _choose_type(self, rng: random.Random) -> str:
        names = list(self.type_weights)
        weights = [self.type_weights[name] for name in names]
        return rng.choices(names, weights=weights, k=1)[0]

    def generate(
        self,
        batch_size: int,
        num_frames: int,
        num_joints: int,
        device: torch.device | str,
        seed: int | None = None,
        force_type: str | None = None,
        force_ratio: float | None = None,
    ) -> tuple[torch.Tensor, list[str], list[float]]:
        rng = random.Random(seed)
        mask = torch.ones(batch_size, num_frames, num_joints, dtype=torch.float32)
        chosen_types: list[str] = []
        chosen_ratios: list[float] = []

        for b in range(batch_size):
            mask_type = force_type or self._choose_type(rng)
            ratio = float(force_ratio if force_ratio is not None else rng.uniform(self.min_ratio, self.max_ratio))
            ratio = min(max(ratio, 1.0 / (num_frames * num_joints)), 0.95)
            chosen_types.append(mask_type)
            chosen_ratios.append(ratio)

            if mask_type == "random_joint":
                count = max(1, round(num_frames * num_joints * ratio))
                ids = rng.sample(range(num_frames * num_joints), min(count, num_frames * num_joints - 1))
                for flat_id in ids:
                    t, j = divmod(flat_id, num_joints)
                    mask[b, t, j] = 0.0

            # elif mask_type == "whole_finger":
            #     group = rng.choice(self.finger_groups)
            #     target = max(1, round(num_frames * len(group) * ratio / max(len(group) / num_joints, 1e-6)))
            #     span = min(num_frames, max(1, math.ceil(target / len(group))))
            #     start = rng.randint(0, num_frames - span)
            #     mask[b, start : start + span, group] = 0.0
            elif mask_type == "whole_finger":
                group = rng.choice(self.finger_groups)
                span = max(
                    1,
                    min(
                        num_frames - 1,
                        round(num_frames * ratio)
                    )
                )
                start = rng.randint(
                    0,
                    num_frames - span
                )
                mask[b, start:start+span, group] = 0.0

            elif mask_type == "temporal_block":
                joint_count = max(1, min(num_joints, round(num_joints * math.sqrt(ratio))))
                span = max(1, min(num_frames, round(num_frames * math.sqrt(ratio))))
                joints = rng.sample(range(num_joints), joint_count)
                start = rng.randint(0, num_frames - span)
                mask[b, start : start + span, joints] = 0.0

            # elif mask_type == "fingertips":
            #     tips = [j for j in self.fingertips if j < num_joints]
            #     span = max(1, min(num_frames, round(num_frames * ratio * num_joints / max(len(tips), 1))))
            #     start = rng.randint(0, num_frames - span)
            #     mask[b, start : start + span, tips] = 0.0
            elif mask_type == "fingertips":
                tips = [
                    j for j in self.fingertips
                    if j < num_joints
                ]
                span = max(
                    1,
                    min(
                        num_frames-1,
                        round(num_frames * ratio)
                    )
                )
                start = rng.randint(
                    0,
                    num_frames-span
                )
                mask[b,start:start+span,tips]=0.0

            elif mask_type == "whole_frame":
                span = max(1, min(num_frames - 1, round(num_frames * ratio)))
                start = rng.randint(0, num_frames - span)
                mask[b, start : start + span, :] = 0.0

            else:
                raise ValueError(f"未知 mask_type: {mask_type}")

            if mask[b].sum() == num_frames * num_joints:
                mask[b, num_frames // 2, num_joints // 2] = 0.0
            if mask[b].sum() == 0:
                mask[b, 0, 0] = 1.0

        return mask.to(device), chosen_types, chosen_ratios

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import nn

from src.model import (
    GeometryEncoder,
    RefinementModule,
    TemporalEncoder,
    make_transformer_encoder,
    normalized_adjacency,
    temporal_linear_fill_torch,
)


@dataclass(frozen=True)
class AblationSpec:
    """一个消融变体只改变一个模型单元或一个损失项。"""

    label: str
    description: str
    use_geometry: bool = True
    use_temporal: bool = True
    use_gate: bool = True
    use_anchor_residual: bool = True
    use_refinement: bool = True
    loss_overrides: Mapping[str, float] = field(default_factory=dict)


ABLATION_SPECS: dict[str, AblationSpec] = {
    "full": AblationSpec(
        label="Full model",
        description="完整 RepairFormer",
    ),
    "no_anchor": AblationSpec(
        label="w/o interpolation prior",
        description="去掉时间线性插值锚点，坐标头直接预测绝对坐标",
        use_anchor_residual=False,
    ),
    "no_geometry": AblationSpec(
        label="w/o geometry branch",
        description="删除局部图几何编码分支",
        use_geometry=False,
    ),
    "no_temporal": AblationSpec(
        label="w/o temporal branch",
        description="删除时序 Transformer 编码分支",
        use_temporal=False,
    ),
    "no_gate": AblationSpec(
        label="w/o adaptive gate",
        description="将自适应门控替换为几何/时序特征固定 0.5 平均",
        use_gate=False,
    ),
    "no_refinement": AblationSpec(
        label="w/o refinement",
        description="删除图结构与局部时间卷积二次细化模块",
        use_refinement=False,
    ),
    "no_velocity_loss": AblationSpec(
        label="w/o velocity loss",
        description="速度监督权重设为 0",
        loss_overrides={"velocity": 0.0},
    ),
    "no_acceleration_loss": AblationSpec(
        label="w/o acceleration loss",
        description="加速度监督权重设为 0",
        loss_overrides={"acceleration": 0.0},
    ),
    "no_structure_loss": AblationSpec(
        label="w/o structural loss",
        description="骨长与骨方向监督权重同时设为 0",
        loss_overrides={"bone": 0.0, "direction": 0.0},
    ),
    "no_consistency_loss": AblationSpec(
        label="w/o velocity consistency",
        description="速度头与坐标差分的一致性监督权重设为 0",
        loss_overrides={"consistency": 0.0},
    ),
    "no_fingertip_weight": AblationSpec(
        label="w/o fingertip weighting",
        description="指尖不再额外加权",
        loss_overrides={"fingertip_weight": 1.0},
    ),
}

MODEL_ABLATIONS = tuple(
    name
    for name, spec in ABLATION_SPECS.items()
    if name == "full"
    or not (
        spec.use_geometry
        and spec.use_temporal
        and spec.use_gate
        and spec.use_anchor_residual
        and spec.use_refinement
    )
)


class AblationRepairFormer(nn.Module):
    """
    当前 RepairFormer 的可拆卸版本。

    结构消融会真正不实例化对应分支，因此参数量统计只包含有效模块；
    不是在完整模型上把输出简单置零。
    """

    def __init__(
        self,
        *,
        num_frames: int,
        num_joints: int,
        edges: list[list[int]],
        d_model: int = 128,
        nhead: int = 8,
        geometry_layers: int = 3,
        temporal_layers: int = 4,
        decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        refinement_layers: int = 2,
        use_geometry: bool = True,
        use_temporal: bool = True,
        use_gate: bool = True,
        use_anchor_residual: bool = True,
        use_refinement: bool = True,
    ) -> None:
        super().__init__()
        if not use_geometry and not use_temporal:
            raise ValueError("geometry 与 temporal 不能同时删除")

        adjacency = normalized_adjacency(num_joints, edges)
        self.num_frames = int(num_frames)
        self.num_joints = int(num_joints)
        self.use_geometry = bool(use_geometry)
        self.use_temporal = bool(use_temporal)
        self.use_gate = bool(use_gate and use_geometry and use_temporal)
        self.use_anchor_residual = bool(use_anchor_residual)
        self.use_refinement = bool(use_refinement)

        self.geometry = (
            GeometryEncoder(adjacency, d_model, geometry_layers, dropout)
            if self.use_geometry
            else None
        )
        self.temporal = (
            TemporalEncoder(
                num_frames,
                d_model,
                nhead,
                temporal_layers,
                dim_feedforward,
                dropout,
            )
            if self.use_temporal
            else None
        )
        self.gate = (
            nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
            if self.use_gate
            else None
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.time_embedding = nn.Parameter(
            torch.zeros(1, num_frames, 1, d_model)
        )
        self.joint_embedding = nn.Parameter(
            torch.zeros(1, 1, num_joints, d_model)
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = make_transformer_encoder(decoder_layer, decoder_layers)
        self.coord_head = nn.Linear(d_model, 3)
        self.velocity_head = nn.Linear(d_model, 3)
        self.refiner = (
            RefinementModule(adjacency, d_model, refinement_layers, dropout)
            if self.use_refinement
            else None
        )

        for parameter in (
            self.mask_token,
            self.time_embedding,
            self.joint_embedding,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)

    def _fuse(
        self,
        masked_coords: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        geometry = (
            self.geometry(masked_coords, visible_mask)
            if self.geometry is not None
            else None
        )
        temporal = (
            self.temporal(masked_coords, visible_mask)
            if self.temporal is not None
            else None
        )

        if geometry is not None and temporal is not None:
            if self.gate is not None:
                gate = self.gate(torch.cat([geometry, temporal], dim=-1))
            else:
                gate = torch.full_like(geometry, 0.5)
            fused = gate * geometry + (1.0 - gate) * temporal
            return fused, gate

        if geometry is not None:
            return geometry, torch.ones_like(geometry)
        if temporal is not None:
            return temporal, torch.zeros_like(temporal)
        raise RuntimeError("没有可用编码分支")

    def forward(
        self,
        masked_coords: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, frames, joints, _ = masked_coords.shape
        if frames > self.num_frames or joints != self.num_joints:
            raise ValueError(
                f"期望 T<={self.num_frames}, V={self.num_joints}，"
                f"实际 T={frames}, V={joints}"
            )

        anchor = temporal_linear_fill_torch(masked_coords, visible_mask)
        fused, gate = self._fuse(masked_coords, visible_mask)

        tokens = torch.where(
            visible_mask.unsqueeze(-1).bool(),
            fused,
            self.mask_token.expand(batch, frames, joints, -1),
        )
        tokens = (
            tokens
            + self.time_embedding[:, :frames]
            + self.joint_embedding[:, :, :joints]
        )
        decoded = self.decoder(
            tokens.reshape(batch, frames * joints, -1)
        ).reshape(batch, frames, joints, -1)

        residual = self.coord_head(decoded)
        coarse = anchor + residual if self.use_anchor_residual else residual
        visible = visible_mask.unsqueeze(-1)
        missing = 1.0 - visible
        coarse_repaired = visible * masked_coords + missing * coarse

        if self.refiner is not None:
            refine_delta = self.refiner(
                coarse_repaired, anchor, visible_mask
            )
            pred_coords = coarse + missing * refine_delta
        else:
            refine_delta = torch.zeros_like(coarse)
            pred_coords = coarse

        pred_velocity = self.velocity_head(
            decoded[:, 1:] - decoded[:, :-1]
        )
        repaired = visible * masked_coords + missing * pred_coords
        return {
            "anchor": anchor,
            "coarse_coords": coarse,
            "refine_delta": refine_delta,
            "pred_coords": pred_coords,
            "pred_velocity": pred_velocity,
            "repaired": repaired,
            "gate": gate,
        }


def get_ablation_spec(name: str) -> AblationSpec:
    key = name.strip().lower()
    if key not in ABLATION_SPECS:
        raise KeyError(
            f"未知消融变体：{name}；可选：{', '.join(ABLATION_SPECS)}"
        )
    return ABLATION_SPECS[key]


def build_ablation_model(
    cfg: dict[str, Any],
    variant: str,
) -> AblationRepairFormer:
    spec = get_ablation_spec(variant)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    return AblationRepairFormer(
        num_frames=int(data_cfg["sequence_length"]),
        num_joints=int(data_cfg["num_joints"]),
        edges=data_cfg["edges"],
        d_model=int(model_cfg.get("d_model", 128)),
        nhead=int(model_cfg.get("nhead", 8)),
        geometry_layers=int(model_cfg.get("geometry_layers", 3)),
        temporal_layers=int(model_cfg.get("temporal_layers", 4)),
        decoder_layers=int(model_cfg.get("decoder_layers", 2)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 256)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        refinement_layers=int(model_cfg.get("refinement_layers", 2)),
        use_geometry=spec.use_geometry,
        use_temporal=spec.use_temporal,
        use_gate=spec.use_gate,
        use_anchor_residual=spec.use_anchor_residual,
        use_refinement=spec.use_refinement,
    )


def loss_config_for_variant(
    base_loss_cfg: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    result = dict(base_loss_cfg)
    result.update(get_ablation_spec(variant).loss_overrides)
    return result


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

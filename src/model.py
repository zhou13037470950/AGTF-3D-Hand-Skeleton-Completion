from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def normalized_adjacency(
    num_joints: int,
    edges: Iterable[Iterable[int]],
) -> torch.Tensor:
    adjacency = torch.eye(num_joints, dtype=torch.float32)
    for i, j in edges:
        i, j = int(i), int(j)
        if 0 <= i < num_joints and 0 <= j < num_joints:
            adjacency[i, j] = adjacency[j, i] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1e-6).pow(-0.5)
    return degree[:, None] * adjacency * degree[None, :]


def make_transformer_encoder(
    layer: nn.TransformerEncoderLayer,
    layers: int,
) -> nn.TransformerEncoder:
    try:
        return nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=layers)


def temporal_linear_fill_torch(
    coords: torch.Tensor,
    visible_mask: torch.Tensor,
) -> torch.Tensor:
    """沿时间轴做线性插值；序列首尾使用最近可见帧填充。"""
    batch, frames, joints, channels = coords.shape
    valid = visible_mask.bool()
    time = torch.arange(frames, device=coords.device).view(1, frames, 1)
    time = time.expand(batch, frames, joints)

    previous = torch.where(valid, time, torch.full_like(time, -1))
    previous = torch.cummax(previous, dim=1).values
    following = torch.where(valid, time, torch.full_like(time, frames))
    following = torch.flip(
        torch.cummin(torch.flip(following, dims=[1]), dim=1).values,
        dims=[1],
    )

    prev_index = previous.clamp(0, frames - 1)
    next_index = following.clamp(0, frames - 1)
    prev_coords = torch.gather(
        coords, 1, prev_index.unsqueeze(-1).expand(-1, -1, -1, channels)
    )
    next_coords = torch.gather(
        coords, 1, next_index.unsqueeze(-1).expand(-1, -1, -1, channels)
    )

    has_prev, has_next = previous >= 0, following < frames
    denominator = (following - previous).clamp_min(1).to(coords.dtype)
    alpha = ((time - previous).to(coords.dtype) / denominator).unsqueeze(-1)
    interpolated = prev_coords + alpha * (next_coords - prev_coords)

    output = torch.zeros_like(coords)
    both = has_prev & has_next
    output = torch.where(both.unsqueeze(-1), interpolated, output)
    output = torch.where((has_prev & ~has_next).unsqueeze(-1), prev_coords, output)
    output = torch.where((~has_prev & has_next).unsqueeze(-1), next_coords, output)
    return torch.where(valid.unsqueeze(-1), coords, output)


class GraphBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        adjacency: torch.Tensor,
        dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("vw,btwc->btvc", self.adjacency, x)
        output = self.linear(propagated)
        output = self.dropout(self.activation(self.norm(output)))
        return output + self.residual(x)


class GeometryEncoder(nn.Module):
    def __init__(
        self,
        adjacency: torch.Tensor,
        d_model: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks, in_dim = [], 4
        for _ in range(layers):
            blocks.append(GraphBlock(in_dim, d_model, adjacency, dropout))
            in_dim = d_model
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        coords: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([coords, visible_mask.unsqueeze(-1)], dim=-1)
        for block in self.blocks:
            x = block(x)
        return x


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        max_frames: int,
        d_model: int,
        nhead: int,
        layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(8, d_model)
        self.time_embedding = nn.Parameter(
            torch.zeros(1, max_frames, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = make_transformer_encoder(layer, layers)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    def forward(
        self,
        coords: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, joints, _ = coords.shape
        velocity = torch.zeros_like(coords)
        velocity[:, 1:] = coords[:, 1:] - coords[:, :-1]

        velocity_mask = torch.ones_like(visible_mask)
        velocity_mask[:, 1:] = visible_mask[:, 1:] * visible_mask[:, :-1]
        features = torch.cat(
            [
                coords,
                velocity,
                visible_mask.unsqueeze(-1),
                velocity_mask.unsqueeze(-1),
            ],
            dim=-1,
        )
        features = features.permute(0, 2, 1, 3).reshape(
            batch * joints, frames, 8
        )
        features = self.input_proj(features) + self.time_embedding[:, :frames]
        encoded = self.encoder(features)
        return encoded.reshape(batch, joints, frames, -1).permute(
            0, 2, 1, 3
        ).contiguous()


class RefinementModule(nn.Module):
    """利用图结构和局部时间卷积修正粗修补结果。"""

    def __init__(
        self,
        adjacency: torch.Tensor,
        d_model: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        graph_blocks, temporal_blocks = [], []
        in_dim = 7
        for _ in range(layers):
            graph_blocks.append(
                GraphBlock(in_dim, d_model, adjacency, dropout)
            )
            temporal_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        d_model,
                        d_model,
                        kernel_size=(3, 1),
                        padding=(1, 0),
                    ),
                    nn.
                    GroupNorm(
                        num_groups=8
                    ,
                        num_channels=d_model
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            in_dim = d_model
        self.graph_blocks = nn.ModuleList(graph_blocks)
        self.temporal_blocks = nn.ModuleList(temporal_blocks)
        self.output = nn.Linear(d_model, 3)

    def forward(
        self,
        coarse_repaired: torch.Tensor,
        anchor: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [coarse_repaired, anchor, visible_mask.unsqueeze(-1)],
            dim=-1,
        )
        for graph, temporal in zip(self.graph_blocks, self.temporal_blocks):
            x = graph(x)
            x = x + temporal(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.output(x)


class RepairFormer(nn.Module):
    """插值先验引导的几何—运动融合骨架修补模型。"""

    def __init__(
        self,
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
        use_anchor_residual: bool = True,
        use_refinement: bool = True,
        refinement_layers: int = 2,
    ) -> None:
        super().__init__()
        adjacency = normalized_adjacency(num_joints, edges)
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.use_anchor_residual = use_anchor_residual
        self.use_refinement = use_refinement

        self.geometry = GeometryEncoder(
            adjacency, d_model, geometry_layers, dropout
        )
        self.temporal = TemporalEncoder(
            num_frames,
            d_model,
            nhead,
            temporal_layers,
            dim_feedforward,
            dropout,
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, 1, d_model)
        )
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
        self.decoder = make_transformer_encoder(
            decoder_layer, decoder_layers
        )
        self.coord_head = nn.Linear(d_model, 3)
        self.velocity_head = nn.Linear(d_model, 3)
        self.refiner = (
            RefinementModule(
                adjacency, d_model, refinement_layers, dropout
            )
            if use_refinement else None
        )

        for parameter in (
            self.mask_token,
            self.time_embedding,
            self.joint_embedding,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)

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
        geometry = self.geometry(masked_coords, visible_mask)
        temporal = self.temporal(masked_coords, visible_mask)
        gate = self.gate(torch.cat([geometry, temporal], dim=-1))
        fused = gate * geometry + (1.0 - gate) * temporal

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
        missing = 1.0 - visible_mask.unsqueeze(-1)
        coarse_repaired = visible_mask.unsqueeze(-1) * masked_coords + missing * coarse

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
        repaired = (
            visible_mask.unsqueeze(-1) * masked_coords
            + missing * pred_coords
        )
        return {
            "anchor": anchor,
            "coarse_coords": coarse,
            "refine_delta": refine_delta,
            "pred_coords": pred_coords,
            "pred_velocity": pred_velocity,
            "repaired": repaired,
            "gate": gate,
        }

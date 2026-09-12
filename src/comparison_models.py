from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .model import GraphBlock, normalized_adjacency


def _velocity(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:] - x[:, :-1]


def _pack_output(
    pred_coords: torch.Tensor,
    masked_coords: torch.Tensor,
    visible_mask: torch.Tensor,
    pred_velocity: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    repaired = (
        visible_mask.unsqueeze(-1) * masked_coords
        + (1.0 - visible_mask.unsqueeze(-1)) * pred_coords
    )
    if pred_velocity is None:
        pred_velocity = _velocity(pred_coords)
    return {
        "pred_coords": pred_coords,
        "pred_velocity": pred_velocity,
        "repaired": repaired,
    }


def temporal_linear_fill_numpy(
    masked_coords: np.ndarray,
    visible_mask: np.ndarray,
) -> np.ndarray:
    x = np.asarray(masked_coords, dtype=np.float32)
    mask = np.asarray(visible_mask, dtype=np.float32)
    output = x.copy()
    _, frames, joints, channels = output.shape
    timeline = np.arange(frames, dtype=np.float32)

    for batch_index in range(output.shape[0]):
        for joint_index in range(joints):
            valid = np.flatnonzero(mask[batch_index, :, joint_index] > 0.5)
            if valid.size == 0:
                output[batch_index, :, joint_index] = 0.0
                continue
            for channel_index in range(channels):
                output[batch_index, :, joint_index, channel_index] = np.interp(
                    timeline,
                    valid.astype(np.float32),
                    x[batch_index, valid, joint_index, channel_index],
                ).astype(np.float32)
    return output


def temporal_linear_fill_torch(
    masked_coords: torch.Tensor,
    visible_mask: torch.Tensor,
) -> torch.Tensor:
    output = temporal_linear_fill_numpy(
        masked_coords.detach().cpu().numpy(),
        visible_mask.detach().cpu().numpy(),
    )
    return torch.from_numpy(output).to(
        device=masked_coords.device,
        dtype=masked_coords.dtype,
    )


class GRUAutoencoder(nn.Module):
    def __init__(
        self,
        num_frames: int,
        num_joints: int,
        d_model: int = 96,
        layers: int = 2,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        del num_frames

        input_dim = num_joints * 4
        hidden_dim = max(32, d_model // 2)

        self.num_joints = num_joints
        self.input_proj = nn.Linear(input_dim, d_model)
        self.encoder = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.output = nn.Linear(hidden_dim * 2, num_joints * 3)

    def forward(
        self,
        masked_coords: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, frames, joints, _ = masked_coords.shape
        features = torch.cat(
            [masked_coords, visible_mask.unsqueeze(-1)],
            dim=-1,
        ).reshape(batch, frames, joints * 4)

        encoded, _ = self.encoder(self.input_proj(features))
        pred = self.output(self.norm(encoded)).reshape(
            batch,
            frames,
            joints,
            3,
        )
        return _pack_output(pred, masked_coords, visible_mask)


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class TCNAutoencoder(nn.Module):
    def __init__(
        self,
        num_frames: int,
        num_joints: int,
        d_model: int = 96,
        layers: int = 4,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        del num_frames

        self.input = nn.Conv1d(num_joints * 4, d_model, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(
                    d_model,
                    2 ** (layer_index % 4),
                    dropout,
                )
                for layer_index in range(layers)
            ]
        )
        self.output = nn.Conv1d(d_model, num_joints * 3, kernel_size=1)

    def forward(
        self,
        masked_coords: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, frames, joints, _ = masked_coords.shape
        features = torch.cat(
            [masked_coords, visible_mask.unsqueeze(-1)],
            dim=-1,
        ).reshape(batch, frames, joints * 4).transpose(1, 2)

        hidden = self.blocks(self.input(features))
        pred = self.output(hidden).transpose(1, 2).reshape(
            batch,
            frames,
            joints,
            3,
        )
        return _pack_output(pred, masked_coords, visible_mask)


class STGCNAutoencoder(nn.Module):
    def __init__(
        self,
        num_frames: int,
        num_joints: int,
        edges: list[list[int]],
        d_model: int = 96,
        layers: int = 3,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        del num_frames

        adjacency = normalized_adjacency(num_joints, edges)
        graph_blocks: list[nn.Module] = []
        temporal_blocks: list[nn.Module] = []
        input_dim = 4

        for _ in range(layers):
            graph_blocks.append(
                GraphBlock(input_dim, d_model, adjacency, dropout)
            )
            temporal_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        d_model,
                        d_model,
                        kernel_size=(3, 1),
                        padding=(1, 0),
                    ),
                    nn.BatchNorm2d(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            input_dim = d_model

        self.graph_blocks = nn.ModuleList(graph_blocks)
        self.temporal_blocks = nn.ModuleList(temporal_blocks)
        self.output = nn.Linear(d_model, 3)

    def forward(
        self,
        masked_coords: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat(
            [masked_coords, visible_mask.unsqueeze(-1)],
            dim=-1,
        )

        for graph_block, temporal_block in zip(
            self.graph_blocks,
            self.temporal_blocks,
        ):
            x = graph_block(x)
            residual = x
            x = temporal_block(
                x.permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)
            x = x + residual

        pred = self.output(x)
        return _pack_output(pred, masked_coords, visible_mask)


class SkeletonTransformer(nn.Module):
    def __init__(
        self,
        num_frames: int,
        num_joints: int,
        d_model: int = 96,
        nhead: int = 4,
        layers: int = 4,
        dim_feedforward: int = 192,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.input_proj = nn.Linear(4, d_model)
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, 1, d_model)
        )
        self.time_embedding = nn.Parameter(
            torch.zeros(1, num_frames, 1, d_model)
        )
        self.joint_embedding = nn.Parameter(
            torch.zeros(1, 1, num_joints, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers,
            )

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)

    def encode(
        self,
        masked_coords: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, joints, _ = masked_coords.shape

        visible_features = self.input_proj(
            torch.cat(
                [masked_coords, visible_mask.unsqueeze(-1)],
                dim=-1,
            )
        )

        tokens = torch.where(
            visible_mask.unsqueeze(-1).bool(),
            visible_features,
            self.mask_token.expand(batch, frames, joints, -1),
        )
        tokens = (
            tokens
            + self.time_embedding[:, :frames]
            + self.joint_embedding[:, :, :joints]
        )

        return self.encoder(
            tokens.reshape(batch, frames * joints, -1)
        ).reshape(batch, frames, joints, -1)


class TransformerMAE(SkeletonTransformer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.coord_head = nn.Linear(
            int(kwargs.get("d_model", 96)),
            3,
        )

    def forward(
        self,
        masked_coords: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred = self.coord_head(
            self.encode(masked_coords, visible_mask)
        )
        return _pack_output(pred, masked_coords, visible_mask)


class MotionMAE(SkeletonTransformer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.velocity_head = nn.Linear(
            int(kwargs.get("d_model", 96)),
            3,
        )

    def forward(
        self,
        masked_coords: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(masked_coords, visible_mask)
        pred_velocity = self.velocity_head(
            hidden[:, 1:] - hidden[:, :-1]
        )
        anchor = temporal_linear_fill_torch(
            masked_coords,
            visible_mask,
        )
        pred_coords = torch.cat(
            [
                anchor[:, :1],
                anchor[:, :1]
                + torch.cumsum(pred_velocity, dim=1),
            ],
            dim=1,
        )
        return _pack_output(
            pred_coords,
            masked_coords,
            visible_mask,
            pred_velocity,
        )


class AnatomyMAE(TransformerMAE):
    anatomy_aware = True


class MacDiff(SkeletonTransformer):
    """
    MacDiff: Masked Conditional Diffusion (ECCV 2024 适配版)
    保留 Transformer 编码器，并加入扩散时间步注入与条件解码
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = kwargs.get("d_model", 96)
        
        # 1. 扩散时间步 (Timestep) 嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 2. 扩散条件解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=kwargs.get("nhead", 4),
            dim_feedforward=kwargs.get("dim_feedforward", 192),
            dropout=kwargs.get("dropout", 0.1),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        num_decoder_layers = kwargs.get("decoder_layers", kwargs.get("layers", 4))
        self.decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        self.coord_head = nn.Linear(d_model, 3)

    def forward(self, masked_coords, visible_mask):
        batch, frames, joints, _ = masked_coords.shape
        device = masked_coords.device
        
        # 1. 编码掩码上下文条件
        encoded = self.encode(masked_coords, visible_mask)
        encoded_flat = encoded.reshape(batch, frames * joints, -1)
        
        # # 2. 模拟扩散采样过程中的正弦时间步编码 (Sinusoidal Timestep Embedding)
        # t = torch.randint(0, 1000, (batch,), device=device).float()
        # 在 forward 中加一个判断
        if self.training:
            t = torch.randint(0, 1000, (batch,), device=device).float()
        else:
            # 推理时使用最后一步（干净状态）
            t = torch.full((batch,), 999, device=device).float()
        half_dim = encoded.shape[-1] // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(np.log(10000.0) / (half_dim - 1)))
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        t_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        t_feat = self.time_mlp(t_emb).unsqueeze(1) # [batch, 1, d_model]
        
        # 3. 将时间步特征注入到 Decoder 条件中
        target_tokens = encoded_flat + t_feat
        decoded_flat = self.decoder(
            tgt=target_tokens, 
            memory=encoded_flat
        )
        
        # 4. 还原形状与坐标预测
        decoded = decoded_flat.reshape(batch, frames, joints, -1)
        pred_coords = self.coord_head(decoded)
        
        return _pack_output(pred_coords, masked_coords, visible_mask)


class AGMAE(SkeletonTransformer):
    """
    AG-MAE: Anatomically Guided Spatio-Temporal MAE (3DV 2025)
    完整实现：解剖图卷积 + 解剖掩码 + 解剖对比损失（全在类内）
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        num_joints = kwargs.get("num_joints", 22)
        edges = kwargs.get("edges", [])
        finger_groups = kwargs.get("finger_groups", [])
        d_model = kwargs.get("d_model", 96)
        dropout = kwargs.get("dropout", 0.1)

        # 保存手指分组供内部使用
        self.finger_groups = finger_groups
        self.num_joints = num_joints

        # 1. 解剖邻接矩阵
        adj = torch.eye(num_joints)
        for i, j in edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        deg = torch.sum(adj, dim=1)
        deg_inv = torch.pow(deg, -0.5)
        deg_inv[torch.isinf(deg_inv)] = 0.0
        adj_norm = torch.diag(deg_inv) @ adj @ torch.diag(deg_inv)
        self.register_buffer("adj_norm", adj_norm)

        # 2. 解剖图嵌入
        self.gcn_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 3. 坐标输出头
        self.coord_head = nn.Linear(d_model, 3)

        # 4. 解剖对比投影头（用于损失）
        self.contrast_proj = nn.Linear(d_model, 64)

    def _generate_anatomy_mask(self, visible_mask: torch.Tensor) -> torch.Tensor:
        """
        生成解剖引导掩码：在可见掩码基础上，额外掩码整根手指
        """
        if not self.finger_groups or self.training is False:
            return visible_mask
        
        batch, frames, joints = visible_mask.shape
        device = visible_mask.device
        
        num_fingers = len(self.finger_groups)
        num_to_mask = max(1, min(num_fingers, torch.randint(1, 3, (1,)).item()))
        
        finger_indices = torch.randperm(num_fingers)[:num_to_mask]
        
        anatomy_mask = visible_mask.clone()
        for idx in finger_indices:
            joints_in_finger = self.finger_groups[idx.item()]
            anatomy_mask[:, :, joints_in_finger] = 0.0
        
        return anatomy_mask

    def _contrast_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        解剖对比损失：同一手指特征相近，不同手指相远
        """
        if not self.finger_groups or self.training is False:
            return features.new_zeros(())
        
        B, T, V, D = features.shape
        device = features.device
        
        proj = self.contrast_proj(features)
        
        centers = []
        for group in self.finger_groups:
            center = proj[:, :, group, :].mean(dim=2)
            centers.append(center)
        centers = torch.stack(centers, dim=2)
        
        centers_norm = centers / (centers.norm(dim=-1, keepdim=True) + 1e-8)
        sim = torch.matmul(centers_norm, centers_norm.transpose(-2, -1))
        
        num_fingers = len(self.finger_groups)
        eye = torch.eye(num_fingers, device=device).unsqueeze(0).unsqueeze(0)
        
        pos = (sim * eye).sum(dim=-1)
        neg = (sim * (1 - eye)).sum(dim=-1) / (num_fingers - 1)
        
        logits = torch.stack([pos, neg], dim=-1) / 0.1
        labels = torch.zeros_like(pos, dtype=torch.long)
        
        loss = F.cross_entropy(
            logits.reshape(-1, 2),
            labels.reshape(-1),
            reduction='mean'
        )
        return loss

    def encode_anatomical(self, masked_coords, visible_mask):
        batch, frames, joints, _ = masked_coords.shape

        # 生成解剖引导掩码
        anatomy_mask = self._generate_anatomy_mask(visible_mask)
        
        # 应用解剖掩码
        masked_anatomy = masked_coords * anatomy_mask.unsqueeze(-1)

        x_in = torch.cat([masked_anatomy, anatomy_mask.unsqueeze(-1)], dim=-1)
        
        # 图卷积
        x_gcn = torch.einsum("ij, btjc -> btic", self.adj_norm, x_in)
        visible_features = self.gcn_proj(x_gcn)

        # Mask Token
        tokens = torch.where(
            anatomy_mask.unsqueeze(-1).bool(),
            visible_features,
            self.mask_token.expand(batch, frames, joints, -1),
        )

        tokens = tokens + self.time_embedding[:, :frames] + self.joint_embedding[:, :, :joints]

        encoded_flat = self.encoder(tokens.reshape(batch, frames * joints, -1))
        return encoded_flat.reshape(batch, frames, joints, -1)

    def forward(self, masked_coords, visible_mask):
        hidden = self.encode_anatomical(masked_coords, visible_mask)
        pred_coords = self.coord_head(hidden)
        
        output = _pack_output(pred_coords, masked_coords, visible_mask)
        
        # 计算解剖对比损失并存入输出
        contrast_loss = self._contrast_loss(hidden)
        output["contrast_loss"] = contrast_loss
        
        return output

class SkeletonInContext(SkeletonTransformer):
    """
    Skeleton-in-Context (CVPR 2024 适配版)
    利用 Support (Prompt Context) 引导 Query 骨架序列的修复
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 不需要额外的 prompt_proj，直接用 input_proj
        self.coord_head = nn.Linear(
            kwargs.get("d_model", 96),
            3,
        )

    def forward(
        self,
        masked_coords: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, frames, joints, _ = masked_coords.shape
        device = masked_coords.device

        # ============================================================
        # 优化后的 Support 构造：从当前序列中提取"可见帧"作为提示
        # 保证训练/测试一致，且 Support 和 Query 有差异（不完全是复制）
        # ============================================================
        
        # 1. 随机选择 20%-50% 的帧作为 Support 帧
        #    用这些帧的可见部分作为上下文
        num_support_frames = max(1, int(frames * torch.randint(2, 5, (1,)).item() / 10))
        
        # 随机选择 Support 帧索引
        support_indices = torch.randperm(frames, device=device)[:num_support_frames]
        
        # 构造 Support：选中的帧保留可见部分，其他帧全部掩码
        support_masked = torch.zeros_like(masked_coords)
        support_mask = torch.zeros_like(visible_mask)
        for idx in support_indices:
            support_masked[:, idx, :, :] = masked_coords[:, idx, :, :]
            support_mask[:, idx, :] = visible_mask[:, idx, :]
        
        # 2. Query 正常使用完整的 masked_coords
        #    Support 和 Query 有差异，但来自同一序列，确保语义连贯
        
        # 3. 特征投影与位置编码
        query_in = torch.cat([masked_coords, visible_mask.unsqueeze(-1)], dim=-1)
        supp_in = torch.cat([support_masked, support_mask.unsqueeze(-1)], dim=-1)

        query_feat = self.input_proj(query_in)
        supp_feat = self.input_proj(supp_in)

        # Mask Token 替换
        query_tokens = torch.where(
            visible_mask.unsqueeze(-1).bool(),
            query_feat,
            self.mask_token.expand(batch, frames, joints, -1),
        )
        supp_tokens = torch.where(
            support_mask.unsqueeze(-1).bool(),
            supp_feat,
            self.mask_token.expand(batch, frames, joints, -1),
        )

        pos_emb = self.time_embedding[:, :frames] + self.joint_embedding[:, :, :joints]
        query_tokens = query_tokens + pos_emb
        supp_tokens = supp_tokens + pos_emb

        # 4. 拼接 Support + Query
        full_tokens = torch.cat([supp_tokens, query_tokens], dim=1)
        full_flat = full_tokens.reshape(batch, 2 * frames * joints, -1)
        encoded_flat = self.encoder(full_flat)
        encoded = encoded_flat.reshape(batch, 2 * frames, joints, -1)

        # 5. 截取 Query 部分并预测坐标
        query_encoded = encoded[:, frames:, :, :]
        pred_coords = self.coord_head(query_encoded)

        return _pack_output(pred_coords, masked_coords, visible_mask)
    

TRAINABLE_MODELS = [
    "gru_ae",
    "tcn_ae",
    "stgcn_ae",
    "transformer_mae",
    "motion_mae",
    "anatomy_mae",
    "ag_mae",      # <--- 添加这一行 (AG-MAE)
    "macdiff",  # <--- 添加这一行
    "sic",  # <--- 添加 Skeleton-in-Context
]


MODEL_DISPLAY_NAMES = {
    "gru_ae": "GRU-AE",
    "tcn_ae": "TCN-AE",
    "stgcn_ae": "ST-GCN-AE",
    "transformer_mae": "Transformer-MAE",
    "motion_mae": "Motion-MAE",
    "anatomy_mae": "Anatomy-MAE",
    "ag_mae": "AG-MAE",  # <--- 添加这一行 (AG-MAE)
    "macdiff": "MacDiff",  # <--- 添加这一行
    "sic": "Skeleton-in-Context",  # <--- 添加映射名称

}


def build_repair_model(
    name: str,
    cfg: dict,
) -> nn.Module:
    name = name.lower()
    if name not in TRAINABLE_MODELS:
        raise ValueError(
            f"未知对比模型：{name}；可选：{TRAINABLE_MODELS}"
        )

    data = cfg["data"]
    params = dict(
        cfg.get("comparison", {}).get("model_params", {})
    )
    common = {
        "num_frames": int(data["sequence_length"]),
        "num_joints": int(data["num_joints"]),
        "edges": data["edges"],
        **params,
    }

    model_classes = {
        "gru_ae": GRUAutoencoder,
        "tcn_ae": TCNAutoencoder,
        "stgcn_ae": STGCNAutoencoder,
        "transformer_mae": TransformerMAE,
        "motion_mae": MotionMAE,
        "anatomy_mae": AnatomyMAE,
        "ag_mae": AGMAE,      # <--- 添加这一行 (AG-MAE)
        "macdiff": MacDiff,  # <--- 添加这一行
        "sic": SkeletonInContext,  # <--- 注册构建函数
    }
    return model_classes[name](**common)

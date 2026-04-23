from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1).unsqueeze(-1)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: str = "relu",
        pool: str = "max",
        se: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        act: nn.Module = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()
        pool_layer: nn.Module = (
            nn.MaxPool2d(2, 2) if pool == "max" else nn.AdaptiveAvgPool2d((4, 4))
        )
        layers: list = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            act,
            pool_layer,
        ]
        if se:
            layers.append(SEBlock(out_ch, reduction=se_reduction))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNEncoder(nn.Module):
    """CNN encoder that outputs BOTH a spatial sequence and a pooled embedding.

    The spatial sequence (B, seq_len, embed_dim) feeds into cross-attention,
    giving attention weights over 16 meaningful spatial positions instead of
    a single scalar — this is what fixes the attention collapse.

    After cross-attention, the sequence is average-pooled back to (B, embed_dim).
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 256,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()

        self.block1 = ConvBlock(in_channels, 32,  activation="relu", pool="max", se=False)
        self.block2 = ConvBlock(32,          64,  activation="relu", pool="max", se=False)
        self.block3 = ConvBlock(64,          128, activation="gelu", pool="max", se=True,
                                se_reduction=se_reduction)
        self.block4 = ConvBlock(128,         256, activation="gelu", pool="adaptive", se=True,
                                se_reduction=se_reduction)

        # Project 256 channels → embed_dim at each spatial position
        # After block4: (B, 256, 4, 4) → seq_len = 4*4 = 16
        self.seq_proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Positional embedding for the 16 spatial positions
        self.pos_emb = nn.Parameter(torch.zeros(1, 16, embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 1, 128, 128)

        Returns
        -------
        pooled : (B, embed_dim)          — for auxiliary classifier heads
        seq    : (B, 16, embed_dim)      — for cross-attention fusion
        """
        x = self.block1(x)   # (B, 32,  64, 64)
        x = self.block2(x)   # (B, 64,  32, 32)
        x = self.block3(x)   # (B, 128, 16, 16)
        x = self.block4(x)   # (B, 256,  4,  4)

        B, C, H, W = x.shape
        # Reshape spatial grid to sequence: (B, H*W, C) = (B, 16, 256)
        seq = x.flatten(2).transpose(1, 2)
        seq = self.seq_proj(seq)          # (B, 16, embed_dim)
        seq = seq + self.pos_emb          # add learnable positional embedding

        pooled = seq.mean(dim=1)          # (B, embed_dim)
        return pooled, seq


def build_encoders(cfg) -> nn.ModuleDict:
    embed_dim    = cfg.model.embed_dim
    se_reduction = cfg.model.encoder.se_reduction
    return nn.ModuleDict({
        "mel":  CNNEncoder(embed_dim=embed_dim, se_reduction=se_reduction),
        "lfcc": CNNEncoder(embed_dim=embed_dim, se_reduction=se_reduction),
        "cqt":  CNNEncoder(embed_dim=embed_dim, se_reduction=se_reduction),
    })
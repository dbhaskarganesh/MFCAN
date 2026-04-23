from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """Cross-attention between two feature sequences.

    Now operates on (B, N, D) sequences where N=16 spatial positions,
    producing (B, N, N) attention weight maps that are actually informative.
    Previously N=1 forced all weights to 1.000.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        query   : (B, N, D)  — sequence being refined
        context : (B, N, D)  — sequence being attended to

        Returns
        -------
        (B, N, D) refined query sequence
        """
        attended, weights = self.attn(query=query, key=context, value=context)
        self.last_attn_weights = weights.detach()   # (B, N_q, N_k) — now meaningful
        return self.norm(query + self.dropout(attended))


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        ff_dim = ff_dim or embed_dim * 4
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class BidirectionalCrossAttentionLayer(nn.Module):
    """One layer of bidirectional cross-attention over three feature sequences."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        streams = ("mel", "lfcc", "cqt")

        self.cross_layers = nn.ModuleDict({
            f"{q}_from_{k}": CrossAttentionLayer(embed_dim, num_heads, dropout)
            for q in streams for k in streams if q != k
        })
        self.merge_norms = nn.ModuleDict({s: nn.LayerNorm(embed_dim) for s in streams})
        self.ffn         = nn.ModuleDict({s: FeedForward(embed_dim, dropout=dropout) for s in streams})

    def forward(
        self,
        mel:  torch.Tensor,    # (B, N, D)
        lfcc: torch.Tensor,    # (B, N, D)
        cqt:  torch.Tensor,    # (B, N, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        streams = {"mel": mel, "lfcc": lfcc, "cqt": cqt}
        refined: Dict[str, torch.Tensor] = {}

        for query_name, query_feat in streams.items():
            cross_sum = torch.zeros_like(query_feat)
            for key_name, key_feat in streams.items():
                if key_name == query_name:
                    continue
                cross_sum = cross_sum + self.cross_layers[f"{query_name}_from_{key_name}"](
                    query_feat, key_feat
                )
            merged = self.merge_norms[query_name](query_feat + cross_sum / 2.0)
            refined[query_name] = self.ffn[query_name](merged)

        return refined["mel"], refined["lfcc"], refined["cqt"]


class CrossAttentionFusion(nn.Module):
    """Stack of BidirectionalCrossAttentionLayers operating on spatial sequences."""

    def __init__(
        self,
        embed_dim:  int = 256,
        num_heads:  int = 8,
        num_layers: int = 2,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            BidirectionalCrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        mel:  torch.Tensor,   # (B, N, D)
        lfcc: torch.Tensor,   # (B, N, D)
        cqt:  torch.Tensor,   # (B, N, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            mel, lfcc, cqt = layer(mel, lfcc, cqt)
        return mel, lfcc, cqt

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        last_layer = self.layers[-1]
        return {
            name: layer.last_attn_weights
            for name, layer in last_layer.cross_layers.items()
            if layer.last_attn_weights is not None
        }
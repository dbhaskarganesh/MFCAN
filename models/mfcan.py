from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.encoders import CNNEncoder
from models.attention import CrossAttentionFusion
from models.losses import AuxClassifier, MFCANLoss


class FusionClassifier(nn.Module):
    def __init__(
        self,
        embed_dim:    int = 256,
        hidden_dims:  Tuple[int, ...] = (512, 128),
        dropout_probs: Tuple[float, ...] = (0.3, 0.2),
        num_classes:  int = 2,
    ) -> None:
        super().__init__()
        in_dim = embed_dim * 3
        layers: list = []
        prev_dim = in_dim
        for h_dim, drop in zip(hidden_dims, dropout_probs):
            layers += [nn.Linear(prev_dim, h_dim), nn.GELU(), nn.Dropout(drop)]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MFCAN(nn.Module):


    def __init__(
        self,
        embed_dim:    int = 256,
        num_heads:    int = 8,
        num_ca_layers: int = 2,
        attn_dropout: float = 0.1,
        se_reduction: int = 16,
        hidden_dims:  Tuple[int, ...] = (512, 128),
        dropout_probs: Tuple[float, ...] = (0.3, 0.2),
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict({
            "mel":  CNNEncoder(embed_dim=embed_dim, se_reduction=se_reduction),
            "lfcc": CNNEncoder(embed_dim=embed_dim, se_reduction=se_reduction),
            "cqt":  CNNEncoder(embed_dim=embed_dim, se_reduction=se_reduction),
        })

        self.fusion = CrossAttentionFusion(
            embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_ca_layers, dropout=attn_dropout,
        )

        self.aux_heads = nn.ModuleDict({
            "mel":  AuxClassifier(embed_dim),
            "lfcc": AuxClassifier(embed_dim),
            "cqt":  AuxClassifier(embed_dim),
        })

        self.classifier = FusionClassifier(
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            dropout_probs=dropout_probs,
        )

        self.embed_dim = embed_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        mel:  torch.Tensor,   # (B, 1, 128, 128)
        lfcc: torch.Tensor,   # (B, 1, 128, 128)
        cqt:  torch.Tensor,   # (B, 1, 128, 128)
    ) -> Dict[str, torch.Tensor]:
        # 1. Encode → pooled (B,D) + sequence (B,16,D)
        pooled_mel,  seq_mel  = self.encoders["mel"](mel)
        pooled_lfcc, seq_lfcc = self.encoders["lfcc"](lfcc)
        pooled_cqt,  seq_cqt  = self.encoders["cqt"](cqt)

        # 2. Auxiliary predictions on pooled features (before cross-attention)
        aux_mel  = self.aux_heads["mel"](pooled_mel)
        aux_lfcc = self.aux_heads["lfcc"](pooled_lfcc)
        aux_cqt  = self.aux_heads["cqt"](pooled_cqt)

        # 3. Bidirectional cross-attention on sequences (B,16,D) → (B,16,D)
        ref_seq_mel, ref_seq_lfcc, ref_seq_cqt = self.fusion(seq_mel, seq_lfcc, seq_cqt)

        # 4. Pool refined sequences back to (B, D)
        ref_mel  = ref_seq_mel.mean(dim=1)
        ref_lfcc = ref_seq_lfcc.mean(dim=1)
        ref_cqt  = ref_seq_cqt.mean(dim=1)

        # 5. Concatenate and classify
        fused  = torch.cat([ref_mel, ref_lfcc, ref_cqt], dim=-1)   # (B, D*3)
        logits = self.classifier(fused)

        return {
            "logits":    logits,
            "aux_mel":   aux_mel,
            "aux_lfcc":  aux_lfcc,
            "aux_cqt":   aux_cqt,
            "emb_mel":   pooled_mel,
            "emb_lfcc":  pooled_lfcc,
            "emb_cqt":   pooled_cqt,
            "emb_fused": fused,
        }

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        return self.fusion.get_attention_weights()

    def forward_single_stream(self, feature: torch.Tensor, stream: str) -> torch.Tensor:
        pooled, _ = self.encoders[stream](feature)
        return self.aux_heads[stream](pooled)

    def forward_two_streams(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        preds = [
            self.aux_heads[k](self.encoders[k](v)[0])
            for k, v in features.items()
        ]
        return torch.stack(preds).mean(0)

    def forward_no_cross_attention(
        self, mel: torch.Tensor, lfcc: torch.Tensor, cqt: torch.Tensor
    ) -> torch.Tensor:
        pooled_mel,  _ = self.encoders["mel"](mel)
        pooled_lfcc, _ = self.encoders["lfcc"](lfcc)
        pooled_cqt,  _ = self.encoders["cqt"](cqt)
        fused = torch.cat([pooled_mel, pooled_lfcc, pooled_cqt], dim=-1)
        return self.classifier(fused)


def build_model(cfg) -> MFCAN:
    m  = cfg.model
    ca = m.cross_attention
    cl = m.classifier
    return MFCAN(
        embed_dim=m.embed_dim,
        num_heads=ca.num_heads,
        num_ca_layers=ca.num_layers,
        attn_dropout=ca.dropout,
        se_reduction=m.encoder.se_reduction,
        hidden_dims=tuple(cl.hidden_dims),
        dropout_probs=tuple(cl.dropout),
    )

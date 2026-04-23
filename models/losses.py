from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxClassifier(nn.Module):
    def __init__(self, embed_dim: int = 256, num_classes: int = 2) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) — down-weights easy examples so the model
    stops over-optimising on the majority class (bonafide) and pays more
    attention to the harder spoof samples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2 is the standard choice. Higher gamma = more focus on hard examples.
    """

    def __init__(
        self,
        gamma:        float = 2.0,
        weight:       Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE gives log(p_t) per sample
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt      = torch.exp(-ce)                      # p_t ∈ (0, 1]
        focal   = (1.0 - pt) ** self.gamma * ce       # up-weight hard examples
        return focal.mean()


class FeatureInconsistencyLoss(nn.Module):
    """Jensen-Shannon divergence across three stream predictions."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    @staticmethod
    def _js_divergence(probs_list: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(probs_list, dim=1)   # (B, n, C)
        mean_p  = stacked.mean(dim=1)              # (B, C)
        jsd = torch.zeros(stacked.size(0), device=stacked.device)
        for p in probs_list:
            kl = F.kl_div(
                torch.log(mean_p + 1e-8), p,
                reduction="none", log_target=False,
            ).sum(dim=-1)
            jsd = jsd + kl
        return jsd / len(probs_list)

    def forward(
        self,
        logits_mel:  torch.Tensor,
        logits_lfcc: torch.Tensor,
        logits_cqt:  torch.Tensor,
    ) -> torch.Tensor:
        p_mel  = F.softmax(logits_mel,  dim=-1)
        p_lfcc = F.softmax(logits_lfcc, dim=-1)
        p_cqt  = F.softmax(logits_cqt,  dim=-1)
        jsd = self._js_divergence([p_mel, p_lfcc, p_cqt])
        return jsd.mean() if self.reduction == "mean" else jsd.sum()


class MFCANLoss(nn.Module):
    """Full training objective.

    L_total = L_main (focal) + L_aux (focal) + lambda * L_inconsistency

    Using Focal Loss instead of CrossEntropy fixes the bonafide-bias by
    reducing the gradient contribution of easy bonafide examples and forcing
    the model to focus on the harder spoof patterns.
    """

    def __init__(
        self,
        inconsistency_weight: float = 0.1,
        label_smoothing:      float = 0.05,
        focal_gamma:          float = 2.0,
        class_weights:        Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.inconsistency_weight = inconsistency_weight
        self.incon_loss = FeatureInconsistencyLoss()

        self.main_loss = FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        self.aux_loss = FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
        )

    def forward(
        self,
        fused_logits:    torch.Tensor,
        aux_mel_logits:  torch.Tensor,
        aux_lfcc_logits: torch.Tensor,
        aux_cqt_logits:  torch.Tensor,
        labels:          torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        l_main = self.main_loss(fused_logits, labels)

        l_aux = (
            self.aux_loss(aux_mel_logits,  labels)
            + self.aux_loss(aux_lfcc_logits, labels)
            + self.aux_loss(aux_cqt_logits,  labels)
        ) / 3.0

        l_incon = self.incon_loss(aux_mel_logits, aux_lfcc_logits, aux_cqt_logits)
        l_total = l_main + l_aux + self.inconsistency_weight * l_incon

        return {
            "total":         l_total,
            "main":          l_main,
            "aux":           l_aux,
            "inconsistency": l_incon,
        }
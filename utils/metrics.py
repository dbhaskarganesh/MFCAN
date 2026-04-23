from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(
    labels: np.ndarray,     
    scores: np.ndarray,     
) -> Tuple[float, float]:


    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    
    abs_diffs = np.abs(fnr - fpr)
    min_idx   = np.argmin(abs_diffs)

    
    if min_idx == 0 or min_idx == len(fpr) - 1:
        eer = (fpr[min_idx] + fnr[min_idx]) / 2.0
        eer_thresh = thresholds[min_idx]
    else:
        
        diff = fpr - fnr
        sign_change = (diff[min_idx] * diff[min_idx - 1]) < 0
        if sign_change:
            t0, t1 = thresholds[min_idx - 1], thresholds[min_idx]
            d0, d1 = abs(diff[min_idx - 1]), abs(diff[min_idx])
            eer_thresh = t0 + (t1 - t0) * d0 / (d0 + d1 + 1e-12)
            eer = (fpr[min_idx] + fnr[min_idx]) / 2.0
        else:
            eer = (fpr[min_idx] + fnr[min_idx]) / 2.0
            eer_thresh = thresholds[min_idx]

    return float(eer), float(eer_thresh)


def compute_min_tdcf(
    labels: np.ndarray,
    scores: np.ndarray,
    Pspoof: float = 0.05,
    Cmiss:  float = 1.0,
    Cfa:    float = 10.0,
) -> float:


    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr   
    far = fpr          

    
    tdcf = Cmiss * Pspoof * fnr + Cfa * (1.0 - Pspoof) * far

    
    
    default_cost = min(Cmiss * Pspoof, Cfa * (1.0 - Pspoof))

    min_tdcf      = float(np.min(tdcf))
    min_tdcf_norm = min_tdcf / (default_cost + 1e-12)

    return min_tdcf_norm


def compute_all_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    tdcf_params: Optional[Dict] = None,
) -> Dict[str, float]:


    if tdcf_params is None:
        tdcf_params = dict(Pspoof=0.05, Cmiss=1.0, Cfa=10.0)

    eer, eer_thresh = compute_eer(labels, scores)
    min_tdcf        = compute_min_tdcf(labels, scores, **tdcf_params)
    auc             = roc_auc_score(labels, scores)

    return {
        "eer_pct":       eer * 100.0,   
        "eer_threshold": eer_thresh,
        "min_tdcf":      min_tdcf,
        "auc_roc":       auc,
    }

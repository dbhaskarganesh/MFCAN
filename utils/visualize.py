from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import sklearn
import torch
from sklearn.manifold import TSNE


def _save_or_show(fig: plt.Figure, path: Optional[str]) -> None:
    if path:
        os.makedirs(Path(path).parent, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def _make_tsne(n_samples: int, random_state: int) -> TSNE:
    sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    iter_kwarg = "max_iter" if sk_version >= (1, 5) else "n_iter"
    perplexity = min(40, max(5, n_samples // 10))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        **{iter_kwarg: 1000},
        random_state=random_state,
    )


def plot_features_comparison(
    real_features: Dict[str, np.ndarray],
    fake_features: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> None:
    feature_names = ["mel", "lfcc", "cqt"]
    titles = ["Mel-Spectrogram", "LFCC (+ Δ + ΔΔ)", "CQT"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Feature Comparison: Real vs Fake", fontsize=14, fontweight="bold")

    for col, (name, title) in enumerate(zip(feature_names, titles)):
        for row, (label, feats) in enumerate(
            [("Bonafide (Real)", real_features), ("Spoof (Fake)", fake_features)]
        ):
            ax = axes[row, col]
            feat = feats[name]
            if feat.ndim == 3:
                feat = feat[0]
            im = ax.imshow(feat, aspect="auto", origin="lower", cmap="magma")
            ax.set_title(f"{title}\n[{label}]", fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _save_or_show(fig, save_path)


def plot_attention_heatmap(
    attn_weights: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
) -> None:
  

    PAIRS = [
        ("mel_from_lfcc", "LFCC → MEL"),
        ("mel_from_cqt",  "CQT  → MEL"),
        ("lfcc_from_mel", "MEL  → LFCC"),
        ("lfcc_from_cqt", "CQT  → LFCC"),
        ("cqt_from_mel",  "MEL  → CQT"),
        ("cqt_from_lfcc", "LFCC → CQT"),
    ]

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#0f0f1a")

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[3, 1],
        hspace=0.35,
    )

  
    top_grid = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[0],
        hspace=0.45, wspace=0.3,
    )


    bot_ax = fig.add_subplot(outer[1])

    fig.suptitle(
        "Cross-Attention Weight Analysis  (averaged over heads)",
        fontsize=15, fontweight="bold", color="white", y=0.97,
    )

    head_entropies: Dict[str, float] = {}

    for idx, (key, title) in enumerate(PAIRS):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(top_grid[row, col])
        ax.set_facecolor("#1a1a2e")

        w = attn_weights.get(key)
        if w is None:
            ax.set_visible(False)
            continue

        
        w_np = w.detach().cpu().float().numpy()

       
        w_np = w_np[0]   # → (heads, N_q, N_k) or (N_q, N_k)

        if w_np.ndim == 3:
            
            eps = 1e-9
            ent = -(w_np * np.log(w_np + eps)).sum(axis=-1).mean(axis=-1)  # (heads,)
            head_entropies[title] = float(ent.mean())
           
            w_avg = w_np.mean(axis=0)   # (N_q, N_k)
        else:
            w_avg = w_np                # (N_q, N_k)
            head_entropies[title] = 0.0


        im = ax.imshow(
            w_avg,
            cmap="viridis",
            vmin=0,
            vmax=w_avg.max(),
            aspect="auto",
            interpolation="nearest",
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=6)
        ax.set_xlabel("Source positions (key)", fontsize=8, color="#aaaacc")
        ax.set_ylabel("Query positions", fontsize=8, color="#aaaacc")

     
        N = w_avg.shape[0]
        tick_step = max(1, N // 4)
        ticks = list(range(0, N, tick_step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    if head_entropies:
        labels  = [t.replace("→", "→\n") for t in head_entropies.keys()]
        heights = list(head_entropies.values())

        colours = plt.cm.plasma(np.linspace(0.3, 0.9, len(labels)))
        bars = bot_ax.bar(labels, heights, color=colours, edgecolor="#222244", linewidth=0.8)
        bot_ax.set_facecolor("#1a1a2e")
        bot_ax.set_title(
            "Attention Head Diversity  (mean entropy per stream pair — higher = more diverse)",
            fontsize=10, color="white", pad=8,
        )
        bot_ax.set_ylabel("Mean entropy", fontsize=9, color="#aaaacc")
        bot_ax.tick_params(axis="x", colors="white", labelsize=8)
        bot_ax.tick_params(axis="y", colors="white", labelsize=8)
        for spine in bot_ax.spines.values():
            spine.set_edgecolor("#444466")
        bot_ax.set_facecolor("#1a1a2e")

        for bar, val in zip(bars, heights):
            bot_ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(heights) * 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, color="white",
            )

    _save_or_show(fig, save_path)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 2000,
    save_path: Optional[str] = None,
    random_state: int = 42,
) -> None:
    if len(embeddings) > n_samples:
        idx = np.random.RandomState(random_state).choice(
            len(embeddings), n_samples, replace=False
        )
        embeddings = embeddings[idx]
        labels = labels[idx]

    tsne   = _make_tsne(len(embeddings), random_state)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))
    colours      = {0: "#e74c3c", 1: "#2ecc71"}
    legend_labels = {0: "Spoof", 1: "Bonafide"}

    for lbl in [0, 1]:
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=colours[lbl], label=legend_labels[lbl],
            alpha=0.55, s=12, linewidths=0,
        )

    ax.set_title("t-SNE of MFCAN Fused Embeddings", fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=3, fontsize=11)
    _save_or_show(fig, save_path)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    if not history:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("MFCAN Training Curves", fontsize=14)

    for key in ["train_loss", "val_loss"]:
        if key in history and len(history[key]) > 0:
            epochs = range(1, len(history[key]) + 1)
            label  = "Train" if "train" in key else "Validation"
            ls     = "-" if "train" in key else "--"
            ax1.plot(epochs, history[key], label=label, linestyle=ls, linewidth=1.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for key in ["train_eer", "val_eer"]:
        if key in history and len(history[key]) > 0:
            epochs = range(1, len(history[key]) + 1)
            label  = "Train" if "train" in key else "Validation"
            ls     = "-" if "train" in key else "--"
            ax2.plot(epochs, history[key], label=label, linestyle=ls, linewidth=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("EER (%)")
    ax2.set_title("Equal Error Rate")
    ax2.legend()
    ax2.grid(alpha=0.3)

    _save_or_show(fig, save_path)


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    eer: float,
    auc: float,
    save_path: Optional[str] = None,
) -> None:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    eer_val = eer / 100.0
    ax.scatter(
        [eer_val], [1 - eer_val], color="#e74c3c",
        zorder=5, s=80, label=f"EER = {eer:.2f}%",
    )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("MFCAN ROC Curve", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save_or_show(fig, save_path)


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    from sklearn.metrics import confusion_matrix

    cm      = confusion_matrix(labels, preds, labels=[0, 1])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".3f", cmap="Blues",
        xticklabels=["Spoof", "Bonafide"],
        yticklabels=["Spoof", "Bonafide"],
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Normalised Confusion Matrix")
    _save_or_show(fig, save_path)

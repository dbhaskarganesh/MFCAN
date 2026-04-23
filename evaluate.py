from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import build_datasets
from models.mfcan import MFCAN, build_model
from utils.metrics import compute_all_metrics
from utils.visualize import (
    plot_attention_heatmap, plot_confusion_matrix,
    plot_roc_curve, plot_tsne,
)


def get_device(cfg) -> torch.device:
    if cfg.project.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.project.device)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


@torch.no_grad()
def collect_scores(model, loader, device, return_embeddings=False) -> Dict:
    """Run inference and return raw scores without applying any threshold."""
    model.eval()
    all_labels:     List[int]   = []
    all_scores:     List[float] = []
    all_embeddings: List[np.ndarray] = []
    last_attn = None

    for batch in tqdm(loader, desc="Scoring", leave=False):
        mel    = batch["mel"].to(device)
        lfcc   = batch["lfcc"].to(device)
        cqt    = batch["cqt"].to(device)
        labels = batch["label"].cpu().numpy()

        outputs   = model(mel, lfcc, cqt)
        log_probs = torch.log_softmax(outputs["logits"], dim=-1)
        scores    = log_probs[:, 1].cpu().numpy()

        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())

        if return_embeddings:
            all_embeddings.append(outputs["emb_fused"].cpu().numpy())

        raw = model.module if hasattr(model, "module") else model
        last_attn = raw.get_attention_weights()

    result = {
        "labels": np.array(all_labels),
        "scores": np.array(all_scores),
    }
    if return_embeddings and all_embeddings:
        result["embeddings"] = np.concatenate(all_embeddings, axis=0)
    if last_attn:
        result["attn_weights"] = last_attn
    return result


def find_best_threshold(labels: np.ndarray, scores: np.ndarray, cfg) -> Tuple[float, float]:
    """Find the threshold on a labelled set that minimises tDCF.

    Returns (best_threshold, best_tdcf).
    Uses direct index alignment between roc_curve outputs — no searchsorted.
    Includes sanity check against degenerate solutions.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr  = 1.0 - tpr

    tdcf = (
        cfg.tdcf.Cmiss * cfg.tdcf.Pspoof       * fnr
        + cfg.tdcf.Cfa * (1.0 - cfg.tdcf.Pspoof) * fpr
    )

    best_idx    = int(np.argmin(tdcf))
    best_thresh = float(thresholds[best_idx])
    best_tdcf   = float(tdcf[best_idx])

    # Sanity check: reject degenerate thresholds
    bonafide_ratio = float((scores >= best_thresh).mean())
    if bonafide_ratio > 0.95 or bonafide_ratio < 0.05:
        print(f"  [Threshold] WARNING: tDCF-optimal threshold {best_thresh:.4f} is degenerate "
              f"({bonafide_ratio:.0%} bonafide). Falling back to EER threshold.")
        eer_idx     = int(np.argmin(np.abs(fnr - fpr)))
        best_thresh = float(thresholds[eer_idx])
        best_tdcf   = float(tdcf[eer_idx])
        bonafide_ratio = float((scores >= best_thresh).mean())

    print(f"  [Threshold] {best_thresh:.4f}  tDCF={best_tdcf:.4f}  "
          f"bonafide_ratio={bonafide_ratio:.1%}")
    return best_thresh, best_tdcf


def full_metrics(labels: np.ndarray, scores: np.ndarray, preds: np.ndarray, cfg) -> Dict:
    base = compute_all_metrics(
        labels, scores,
        tdcf_params=dict(Pspoof=cfg.tdcf.Pspoof, Cmiss=cfg.tdcf.Cmiss, Cfa=cfg.tdcf.Cfa),
    )
    base["accuracy"]           = float(accuracy_score(labels, preds))
    base["precision_macro"]    = float(precision_score(labels, preds, average="macro",    zero_division=0))
    base["recall_macro"]       = float(recall_score(labels,    preds, average="macro",    zero_division=0))
    base["f1_macro"]           = float(f1_score(labels,        preds, average="macro",    zero_division=0))
    base["precision_weighted"] = float(precision_score(labels, preds, average="weighted", zero_division=0))
    base["recall_weighted"]    = float(recall_score(labels,    preds, average="weighted", zero_division=0))
    base["f1_weighted"]        = float(f1_score(labels,        preds, average="weighted", zero_division=0))

    for cls_idx, cls_name in [(0, "spoof"), (1, "bonafide")]:
        base[f"precision_{cls_name}"] = float(precision_score(labels, preds, labels=[cls_idx], average="micro", zero_division=0))
        base[f"recall_{cls_name}"]    = float(recall_score(labels,    preds, labels=[cls_idx], average="micro", zero_division=0))
        base[f"f1_{cls_name}"]        = float(f1_score(labels,        preds, labels=[cls_idx], average="micro", zero_division=0))
        base[f"n_samples_{cls_name}"] = int((labels == cls_idx).sum())

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    base["confusion_matrix"] = {
        "true_negative":  int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive":  int(tp),
        "matrix": cm.tolist(), "labels": ["spoof(0)", "bonafide(1)"],
    }
    base["n_total_samples"] = int(len(labels))
    return base


def build_paper_report(metrics, ablation_results, ckpt, cfg, model,
                        checkpoint_path, eval_set_size, threshold) -> Dict:
    report = {
        "experiment": {
            "title":                "MFCAN: Multi-Feature Cross-Attention Network for Audio Deepfake Detection",
            "dataset":              "ASVspoof 2019 Logical Access (LA)",
            "timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_path":      str(checkpoint_path),
            "best_epoch":           ckpt.get("epoch", "unknown"),
            "val_eer_at_best":      round(float(ckpt.get("val_eer",  -1)), 4),
            "val_tdcf_at_best":     round(float(ckpt.get("val_tdcf", -1)), 4),
            "calibrated_threshold": round(float(threshold), 6),
            "threshold_source":     "dev-set tDCF minimisation (re-calibrated at eval time)",
        },
        "model_info": {
            "architecture":           "MFCAN v2",
            "features":               ["Mel-Spectrogram", "LFCC (60+delta+delta-delta)", "CQT (84 bins)"],
            "embed_dim":              cfg.model.embed_dim,
            "cross_attention_heads":  cfg.model.cross_attention.num_heads,
            "cross_attention_layers": cfg.model.cross_attention.num_layers,
            "attention_seq_len":      16,
            "se_reduction":           cfg.model.encoder.se_reduction,
            **count_parameters(model),
        },
        "training_config": {
            "epochs_trained":          ckpt.get("epoch", "unknown"),
            "max_epochs":              cfg.training.epochs,
            "batch_size":              cfg.training.batch_size,
            "optimizer":               cfg.training.optimizer.name,
            "learning_rate":           cfg.training.optimizer.lr,
            "weight_decay":            cfg.training.optimizer.weight_decay,
            "scheduler":               cfg.training.scheduler.name,
            "early_stopping_patience": cfg.training.early_stopping.patience,
            "loss_function":           cfg.loss.type,
            "focal_gamma":             cfg.loss.focal_gamma,
            "inconsistency_loss_weight": cfg.loss.inconsistency_weight,
        },
        "primary_metrics": {
            "EER_%":    round(metrics["eer_pct"],  4),
            "min_tDCF": round(metrics["min_tdcf"], 4),
            "AUC_ROC":  round(metrics["auc_roc"],  4),
            "accuracy": round(metrics["accuracy"], 4),
        },
        "detailed_metrics": {
            k: round(v, 4) for k, v in metrics.items()
            if isinstance(v, float) and k != "eer_threshold"
        },
        "per_class_metrics": {
            "spoof":    {"precision": round(metrics["precision_spoof"],    4),
                         "recall":    round(metrics["recall_spoof"],       4),
                         "f1":        round(metrics["f1_spoof"],           4),
                         "n_samples": metrics["n_samples_spoof"]},
            "bonafide": {"precision": round(metrics["precision_bonafide"], 4),
                         "recall":    round(metrics["recall_bonafide"],    4),
                         "f1":        round(metrics["f1_bonafide"],        4),
                         "n_samples": metrics["n_samples_bonafide"]},
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "dataset_stats": {
            "eval_set_total": eval_set_size,
            "n_spoof":        metrics["n_samples_spoof"],
            "n_bonafide":     metrics["n_samples_bonafide"],
            "spoof_ratio_%":  round(metrics["n_samples_spoof"] / max(eval_set_size, 1) * 100, 2),
        },
        "tdcf_cost_params": {"Pspoof": cfg.tdcf.Pspoof, "Cmiss": cfg.tdcf.Cmiss, "Cfa": cfg.tdcf.Cfa},
        "environment": {
            "python_version":  platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available":  torch.cuda.is_available(),
            "cuda_device":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
    }

    if ablation_results:
        report["ablation_study"] = {
            cond: {
                "EER_%":    round(m["eer_pct"],  4),
                "min_tDCF": round(m["min_tdcf"], 4),
                "AUC_ROC":  round(m["auc_roc"],  4),
                "accuracy": round(m["accuracy"], 4),
                "f1_macro": round(m["f1_macro"], 4),
            }
            for cond, m in ablation_results.items()
        }
        best = min(ablation_results, key=lambda c: ablation_results[c]["eer_pct"])
        report["ablation_study"]["_summary"] = {
            "best_condition":     best,
            "best_eer_%":         round(ablation_results[best]["eer_pct"], 4),
            "full_model_eer_%":   round(ablation_results.get("full_mfcan", {}).get("eer_pct", -1), 4),
            "gain_over_only_mel": round(
                ablation_results.get("only_mel",  {}).get("eer_pct", 0)
                - ablation_results.get("full_mfcan", {}).get("eer_pct", 0), 4
            ),
        }
    return report


ABLATION_CONDITIONS = [
    "only_mel", "only_lfcc", "only_cqt",
    "mel_lfcc", "mel_cqt",   "lfcc_cqt",
    "no_cross_attention", "full_mfcan",
]


@torch.no_grad()
def run_ablation_condition(model, loader, device, condition, cfg, threshold) -> Dict:
    model.eval()
    all_labels: List[int]   = []
    all_scores: List[float] = []

    for batch in tqdm(loader, desc=condition, leave=False):
        mel    = batch["mel"].to(device)
        lfcc   = batch["lfcc"].to(device)
        cqt    = batch["cqt"].to(device)
        labels = batch["label"].cpu().numpy()

        if condition == "only_mel":
            logits = model.forward_single_stream(mel, "mel")
        elif condition == "only_lfcc":
            logits = model.forward_single_stream(lfcc, "lfcc")
        elif condition == "only_cqt":
            logits = model.forward_single_stream(cqt, "cqt")
        elif condition == "mel_lfcc":
            logits = model.forward_two_streams({"mel": mel, "lfcc": lfcc})
        elif condition == "mel_cqt":
            logits = model.forward_two_streams({"mel": mel, "cqt": cqt})
        elif condition == "lfcc_cqt":
            logits = model.forward_two_streams({"lfcc": lfcc, "cqt": cqt})
        elif condition == "no_cross_attention":
            logits = model.forward_no_cross_attention(mel, lfcc, cqt)
        else:
            logits = model(mel, lfcc, cqt)["logits"]

        scores = torch.log_softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())

    labels_np = np.array(all_labels)
    scores_np  = np.array(all_scores)
    preds      = (scores_np >= threshold).astype(int)
    return full_metrics(labels_np, scores_np, preds, cfg)


def print_ablation_table(results: Dict) -> None:
    header = f"{'Condition':<25} {'EER (%)':>8} {'min-tDCF':>10} {'AUC-ROC':>9} {'Acc':>7} {'F1':>7}"
    print("\n" + "=" * 72)
    print("ABLATION STUDY RESULTS")
    print("=" * 72)
    print(header)
    print("-" * 72)
    for cond, m in results.items():
        print(
            f"{cond:<25} {m['eer_pct']:>8.2f} {m['min_tdcf']:>10.4f}"
            f" {m['auc_roc']:>9.4f} {m['accuracy']:>7.4f} {m['f1_macro']:>7.4f}"
        )
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MFCAN")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out",        default=None)
    parser.add_argument("--ablation",   action="store_true")
    parser.add_argument("--no-plots",   action="store_true")
    args = parser.parse_args()

    cfg       = OmegaConf.load(args.config)
    ckpt_path = args.checkpoint or cfg.evaluation.checkpoint
    out_dir   = Path(args.out or cfg.evaluation.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg)
    print(f"[Eval] device={device}  checkpoint={ckpt_path}")

    model = build_model(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[Eval] Loaded checkpoint — epoch={ckpt.get('epoch','?')}  "
          f"val_eer={ckpt.get('val_eer','?'):.4f}")

    # Always load dev set for threshold re-calibration at eval time
    # This guarantees a valid threshold even if the saved one is degenerate
    _, dev_ds, eval_ds = build_datasets(cfg)

    dev_loader = DataLoader(
        dev_ds, batch_size=cfg.training.batch_size * 2,
        shuffle=False, num_workers=cfg.data.num_workers,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=cfg.training.batch_size * 2,
        shuffle=False, num_workers=cfg.data.num_workers,
    )

    # Re-calibrate threshold on dev set — overrides the saved checkpoint value
    print("[Eval] Re-calibrating threshold on dev set ...")
    dev_result = collect_scores(model, dev_loader, device)
    threshold, _ = find_best_threshold(dev_result["labels"], dev_result["scores"], cfg)

    saved_thresh = ckpt.get("calibrated_threshold", None)
    if saved_thresh is not None:
        print(f"  Saved threshold was: {saved_thresh:.4f}  →  Using re-calibrated: {threshold:.4f}")

    # Collect eval set scores and apply calibrated threshold
    print("[Eval] Running inference on eval set ...")
    result  = collect_scores(model, eval_loader, device, return_embeddings=True)
    preds   = (result["scores"] >= threshold).astype(int)
    metrics = full_metrics(result["labels"], result["scores"], preds, cfg)

    print("\n" + "=" * 55)
    print("MFCAN — PRIMARY METRICS")
    print("=" * 55)
    for name, val in [
        ("EER (%)",           metrics["eer_pct"]),
        ("min-tDCF",          metrics["min_tdcf"]),
        ("AUC-ROC",           metrics["auc_roc"]),
        ("Accuracy",          metrics["accuracy"]),
        ("F1 (macro)",        metrics["f1_macro"]),
        ("Recall spoof",      metrics["recall_spoof"]),
        ("Precision spoof",   metrics["precision_spoof"]),
    ]:
        print(f"  {name:<25}: {val:.4f}")
    print("=" * 55 + "\n")

    ablation_results = None
    if args.ablation:
        print("[Eval] Running ablation study ...")
        ablation_results = {}
        for cond in ABLATION_CONDITIONS:
            print(f"  ▶ {cond}")
            ablation_results[cond] = run_ablation_condition(
                model, eval_loader, device, cond, cfg, threshold
            )
        print_ablation_table(ablation_results)
        with open(out_dir / "ablation_results.json", "w") as f:
            json.dump(ablation_results, f, indent=2)

    paper_report = build_paper_report(
        metrics, ablation_results, ckpt, cfg, model, ckpt_path, len(eval_ds), threshold,
    )
    with open(out_dir / "paper_results.json", "w") as f:
        json.dump(paper_report, f, indent=2)

    serialisable = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    serialisable["confusion_matrix"] = metrics["confusion_matrix"]
    with open(out_dir / "metrics_raw.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    if not args.no_plots:
        print("[Eval] Generating plots ...")
        plot_roc_curve(
            result["labels"], result["scores"],
            eer=metrics["eer_pct"], auc=metrics["auc_roc"],
            save_path=str(out_dir / "roc_curve.png"),
        )
        plot_confusion_matrix(
            result["labels"], preds,
            save_path=str(out_dir / "confusion_matrix.png"),
        )
        if "embeddings" in result:
            plot_tsne(
                result["embeddings"], result["labels"],
                n_samples=cfg.evaluation.tsne_n_samples,
                save_path=str(out_dir / "tsne.png"),
            )
        if "attn_weights" in result:
            plot_attention_heatmap(
                result["attn_weights"],
                save_path=str(out_dir / "attention_heatmap.png"),
            )
        history_path = Path(cfg.training.save_dir) / "history.json"
        if history_path.exists():
            from utils.visualize import plot_training_curves
            with open(history_path) as f:
                history = json.load(f)
            plot_training_curves(history, save_path=str(out_dir / "training_curves.png"))

    print(f"\n[Eval] All outputs saved to {out_dir}")
    print(f"  paper_results.json")
    print(f"  metrics_raw.json")
    if args.ablation:
        print(f"  ablation_results.json")


if __name__ == "__main__":
    main()
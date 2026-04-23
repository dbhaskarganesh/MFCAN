from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import build_datasets
from models.mfcan import MFCAN, build_model
from models.losses import MFCANLoss
from utils.logger import TrainingLogger
from utils.metrics import compute_all_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(cfg) -> torch.device:
    if cfg.project.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.project.device)


def batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, cfg, logger, epoch, global_step
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train E{epoch}", leave=False)):
        batch  = batch_to_device(batch, device)
        mel, lfcc, cqt, labels = batch["mel"], batch["lfcc"], batch["cqt"], batch["label"]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.project.use_amp):
            outputs   = model(mel, lfcc, cqt)
            loss_dict = criterion(
                outputs["logits"], outputs["aux_mel"],
                outputs["aux_lfcc"], outputs["aux_cqt"], labels,
            )

        scaler.scale(loss_dict["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss  += loss_dict["total"].item()
        global_step += 1

        if batch_idx % cfg.logging.log_interval == 0:
            logger.log_batch(global_step, {
                "batch/loss_total":         loss_dict["total"].item(),
                "batch/loss_main":          loss_dict["main"].item(),
                "batch/loss_inconsistency": loss_dict["inconsistency"].item(),
            })

    return total_loss / len(loader), global_step


@torch.no_grad()
def evaluate(model, loader, device, cfg) -> Dict[str, float]:
    model.eval()
    all_scores: List[float] = []
    all_labels: List[int]   = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        batch  = batch_to_device(batch, device)
        mel, lfcc, cqt, labels = batch["mel"], batch["lfcc"], batch["cqt"], batch["label"]
        outputs   = model(mel, lfcc, cqt)
        log_probs = torch.log_softmax(outputs["logits"], dim=-1)
        all_scores.extend(log_probs[:, 1].cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return compute_all_metrics(
        np.array(all_labels), np.array(all_scores),
        tdcf_params=dict(Pspoof=cfg.tdcf.Pspoof, Cmiss=cfg.tdcf.Cmiss, Cfa=cfg.tdcf.Cfa),
    )


@torch.no_grad()
def calibrate_threshold(model, dev_loader, device, cfg) -> float:
    
    from sklearn.metrics import roc_curve

    model.eval()
    all_scores: List[float] = []
    all_labels: List[int]   = []

    for batch in tqdm(dev_loader, desc="Calibrate", leave=False):
        batch  = batch_to_device(batch, device)
        mel, lfcc, cqt, labels = batch["mel"], batch["lfcc"], batch["cqt"], batch["label"]
        outputs   = model(mel, lfcc, cqt)
        log_probs = torch.log_softmax(outputs["logits"], dim=-1)
        all_scores.extend(log_probs[:, 1].cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    labels_np = np.array(all_labels)
    scores_np  = np.array(all_scores)

    fpr, tpr, thresholds = roc_curve(labels_np, scores_np, pos_label=1)
    fnr  = 1.0 - tpr

    # fpr[i], fnr[i], thresholds[i] are perfectly aligned — direct indexing
    tdcf = (
        cfg.tdcf.Cmiss * cfg.tdcf.Pspoof       * fnr
        + cfg.tdcf.Cfa * (1.0 - cfg.tdcf.Pspoof) * fpr
    )

    best_idx    = int(np.argmin(tdcf))
    best_thresh = float(thresholds[best_idx])
    best_tdcf   = float(tdcf[best_idx])

    # Sanity check: reject if threshold predicts >95% as one class (degenerate)
    bonafide_ratio = float((scores_np >= best_thresh).mean())
    if bonafide_ratio > 0.95 or bonafide_ratio < 0.05:
        print(f"  [Calibration] WARNING: degenerate threshold {best_thresh:.4f} "
              f"({bonafide_ratio:.0%} bonafide) — falling back to EER threshold")
        eer_idx     = int(np.argmin(np.abs(fnr - fpr)))
        best_thresh = float(thresholds[eer_idx])
        best_tdcf   = float(tdcf[eer_idx])
        bonafide_ratio = float((scores_np >= best_thresh).mean())

    print(f"  [Calibration] threshold={best_thresh:.4f}  "
          f"dev_tDCF={best_tdcf:.4f}  bonafide_ratio={bonafide_ratio:.1%}")
    return best_thresh


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "min") -> None:
        self.patience  = patience
        self.mode      = mode
        self.counter   = 0
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.improved  = False

    def step(self, value: float) -> bool:
        improved = (
            (self.mode == "min" and value < self.best) or
            (self.mode == "max" and value > self.best)
        )
        if improved:
            self.best     = value
            self.counter  = 0
            self.improved = True
        else:
            self.counter  += 1
            self.improved  = False
        return self.counter >= self.patience


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MFCAN")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume",   default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.no_wandb:
        cfg.logging.use_wandb = False

    set_seed(cfg.project.seed)
    device = get_device(cfg)
    print(f"[MFCAN] device={device}")

    train_ds, dev_ds, _ = build_datasets(cfg)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=cfg.training.batch_size * 2, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )

    model = build_model(cfg).to(device)
    if cfg.project.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    class_weights = train_ds.compute_class_weights().to(device)
    criterion = MFCANLoss(
        inconsistency_weight=cfg.loss.inconsistency_weight,
        label_smoothing=cfg.loss.label_smoothing,
        focal_gamma=cfg.loss.focal_gamma,
        class_weights=class_weights,
    )

    opt_cfg   = cfg.training.optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt_cfg.lr,
        weight_decay=opt_cfg.weight_decay, betas=tuple(opt_cfg.betas),
    )
    sch_cfg   = cfg.training.scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=sch_cfg.T_0, T_mult=sch_cfg.T_mult, eta_min=sch_cfg.eta_min,
    )
    scaler = GradScaler(enabled=cfg.project.use_amp)

    logger = TrainingLogger(
        save_dir=cfg.training.save_dir,
        use_wandb=cfg.logging.use_wandb,
        wandb_cfg={
            "project": cfg.logging.wandb_project,
            "entity":  cfg.logging.wandb_entity,
            "config":  OmegaConf.to_container(cfg, resolve=True),
            "name":    f"MFCAN_v2_{cfg.project.seed}",
        },
    )

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[MFCAN] Resumed from epoch {ckpt['epoch']}")

    early_stop = EarlyStopping(patience=cfg.training.early_stopping.patience, mode="min")
    save_dir   = Path(cfg.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history: Dict = {"train_loss": [], "val_eer": [], "val_tdcf": []}
    global_step = (start_epoch - 1) * len(train_loader)

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, cfg, logger, epoch, global_step,
        )
        val_metrics = evaluate(model, dev_loader, device, cfg)
        scheduler.step()

        logger.log(epoch, {
            "train/loss":   train_loss,
            "val/eer_pct":  val_metrics["eer_pct"],
            "val/min_tdcf": val_metrics["min_tdcf"],
            "val/auc_roc":  val_metrics["auc_roc"],
            "lr":           optimizer.param_groups[0]["lr"],
        }, step=global_step)

        history["train_loss"].append(train_loss)
        history["val_eer"].append(val_metrics["eer_pct"])
        history["val_tdcf"].append(val_metrics["min_tdcf"])

        early_stop.step(val_metrics["eer_pct"])

        if early_stop.improved:
            calibrated_thresh = calibrate_threshold(model, dev_loader, device, cfg)
            torch.save({
                "epoch":                epoch,
                "model":                model.state_dict(),
                "optimizer":            optimizer.state_dict(),
                "scheduler":            scheduler.state_dict(),
                "val_eer":              val_metrics["eer_pct"],
                "val_tdcf":             val_metrics["min_tdcf"],
                "calibrated_threshold": calibrated_thresh,
                "cfg":                  OmegaConf.to_container(cfg, resolve=True),
            }, save_dir / cfg.training.best_model_name)
            print(f"  ✓ Best saved  EER={val_metrics['eer_pct']:.2f}%  "
                  f"tDCF={val_metrics['min_tdcf']:.4f}  thresh={calibrated_thresh:.4f}")

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        }, save_dir / cfg.training.last_model_name)

        if early_stop.counter >= cfg.training.early_stopping.patience:
            print(f"[MFCAN] Early stopping at epoch {epoch}")
            break

    logger.finish()
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("[MFCAN] Training complete.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from rich.console import Console
from rich.table import Table


console = Console()


class TrainingLogger:


    def __init__(
        self,
        save_dir: str = "./checkpoints",
        use_wandb: bool = False,
        wandb_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.save_dir / "mfcan_log.csv"
        self._csv_header_written = False
        self._wandb_run = None

        if use_wandb and _WANDB_AVAILABLE:
            cfg = wandb_cfg or {}
            self._wandb_run = wandb.init(**cfg)
            console.print("[bold green]WandB run initialised.[/bold green]")
        elif use_wandb and not _WANDB_AVAILABLE:
            console.print("[yellow]wandb not installed — skipping WandB logging.[/yellow]")

    
    
    

    def log(self, epoch: int, metrics: Dict[str, float], step: Optional[int] = None) -> None:


        row = {"epoch": epoch, "timestamp": datetime.now().strftime("%H:%M:%S"), **metrics}
        self._write_csv(row)
        self._print_table(epoch, metrics)
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step or epoch)

    def log_batch(self, step: int, metrics: Dict[str, float]) -> None:

        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def finish(self) -> None:

        if self._wandb_run is not None:
            self._wandb_run.finish()
            console.print("[bold green]WandB run finished.[/bold green]")

    
    
    

    def _write_csv(self, row: Dict[str, Any]) -> None:
        mode = "a" if self._csv_header_written else "w"
        with open(self.csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)

    def _print_table(self, epoch: int, metrics: Dict[str, float]) -> None:
        table = Table(title=f"Epoch {epoch}", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Value",  justify="right")
        for k, v in metrics.items():
            table.add_row(k, f"{v:.6f}" if isinstance(v, float) else str(v))
        console.print(table)

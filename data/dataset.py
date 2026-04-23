from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocess import FeatureExtractor


LABEL_MAP: Dict[str, int] = {"bonafide": 1, "spoof": 0}


def parse_protocol(protocol_path: str) -> List[Dict]:


    records = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id, utt_id, _, system_id, label_str = parts[:5]
            records.append({
                "speaker_id": speaker_id,
                "utt_id":     utt_id,
                "system_id":  system_id,
                "label_str":  label_str,
                "label":      LABEL_MAP[label_str],
            })
    return records


class ASVspoofDataset(Dataset):


    def __init__(
        self,
        root_dir: str,
        protocol_path: str,
        extractor: FeatureExtractor,
        subset: str = "train",
        cache: bool = False,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.cache = cache
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        self.records = parse_protocol(protocol_path)

        
        
        self.flac_dir = Path(root_dir) / f"ASVspoof2019_LA_{subset}" / "flac"
        if not self.flac_dir.exists():
            
            self.flac_dir = Path(root_dir) / "flac"

    
    
    

    def __len__(self) -> int:
        return len(self.records)

    def get_labels(self) -> List[int]:

        return [r["label"] for r in self.records]

    def compute_class_weights(self) -> torch.Tensor:


        labels = np.array(self.get_labels())
        n_total  = len(labels)
        n_spoof   = (labels == 0).sum()
        n_bonafide = (labels == 1).sum()
        
        w_spoof    = n_total / (2.0 * n_spoof)
        w_bonafide = n_total / (2.0 * n_bonafide)
        return torch.tensor([w_spoof, w_bonafide], dtype=torch.float32)

    
    
    

    def _load_audio(self, utt_id: str) -> np.ndarray:

        path = self.flac_dir / f"{utt_id}.flac"
        waveform, _ = librosa.load(
            path,
            sr=self.extractor.sr,
            mono=True,
            dtype=np.float32,
        )
        return waveform

    
    
    

    def __getitem__(self, idx: int) -> Dict:


        if self.cache and idx in self._cache:
            mel, lfcc, cqt = self._cache[idx]
        else:
            rec = self.records[idx]
            waveform = self._load_audio(rec["utt_id"])
            mel, lfcc, cqt = self.extractor(waveform)
            if self.cache:
                self._cache[idx] = (mel, lfcc, cqt)

        rec = self.records[idx]
        return {
            "mel":       mel,
            "lfcc":      lfcc,
            "cqt":       cqt,
            "label":     torch.tensor(rec["label"], dtype=torch.long),
            "utt_id":    rec["utt_id"],
            "system_id": rec["system_id"],
        }


def build_datasets(cfg) -> Tuple[ASVspoofDataset, ASVspoofDataset, ASVspoofDataset]:


    from data.preprocess import FeatureExtractor  

    train_extractor = FeatureExtractor(cfg, augment=True)
    infer_extractor = FeatureExtractor(cfg, augment=False)

    proto_dir = cfg.data.protocol_dir

    train_ds = ASVspoofDataset(
        root_dir=cfg.data.root_dir,
        protocol_path=os.path.join(proto_dir, cfg.data.train_protocol),
        extractor=train_extractor,
        subset="train",
    )
    dev_ds = ASVspoofDataset(
        root_dir=cfg.data.root_dir,
        protocol_path=os.path.join(proto_dir, cfg.data.dev_protocol),
        extractor=infer_extractor,
        subset="dev",
    )
    eval_ds = ASVspoofDataset(
        root_dir=cfg.data.root_dir,
        protocol_path=os.path.join(proto_dir, cfg.data.eval_protocol),
        extractor=infer_extractor,
        subset="eval",
    )
    return train_ds, dev_ds, eval_ds

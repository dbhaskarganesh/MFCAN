from __future__ import annotations

import random
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F


def _to_db(S: np.ndarray, top_db: float = 80.0) -> np.ndarray:

    S_db = librosa.power_to_db(S, ref=np.max, top_db=top_db)
    S_norm = (S_db + top_db) / top_db          
    return S_norm.astype(np.float32)


def _resize(feature: np.ndarray, target: Tuple[int, int] = (128, 128)) -> torch.Tensor:


    t = torch.from_numpy(feature).unsqueeze(0).unsqueeze(0)   
    t = F.interpolate(t, size=target, mode="bilinear", align_corners=False)
    return t.squeeze(0)   


def _fix_length(waveform: np.ndarray, sr: int, max_duration: float = 4.0) -> np.ndarray:

    target_len = int(sr * max_duration)
    if len(waveform) >= target_len:
        return waveform[:target_len]
    return np.pad(waveform, (0, target_len - len(waveform)))


def extract_mel(
    waveform: np.ndarray,
    sr: int = 16_000,
    n_mels: int = 128,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
    fmax: float = 8_000.0,
    power: float = 2.0,
    top_db: float = 80.0,
    output_size: Tuple[int, int] = (128, 128),
) -> torch.Tensor:


    S = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        fmax=fmax,
        power=power,
    )
    S_norm = _to_db(S, top_db=top_db)
    return _resize(S_norm, output_size)


def _linear_filterbank(sr: int, n_fft: int, n_filter: int) -> np.ndarray:


    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    n_bins = len(freqs)
    filter_freqs = np.linspace(0, sr / 2, n_filter + 2)
    fb = np.zeros((n_filter, n_bins), dtype=np.float32)
    for m in range(1, n_filter + 1):
        f_m_minus = filter_freqs[m - 1]
        f_m       = filter_freqs[m]
        f_m_plus  = filter_freqs[m + 1]
        left  = (freqs >= f_m_minus) & (freqs <= f_m)
        right = (freqs >= f_m)       & (freqs <= f_m_plus)
        fb[m - 1, left]  = (freqs[left]  - f_m_minus) / (f_m - f_m_minus + 1e-8)
        fb[m - 1, right] = (f_m_plus - freqs[right]) / (f_m_plus - f_m   + 1e-8)
    return fb


def extract_lfcc(
    waveform: np.ndarray,
    sr: int = 16_000,
    n_filter: int = 70,
    n_lfcc: int = 60,
    n_fft: int = 512,
    win_length: int = 512,
    hop_length: int = 256,
    with_delta: bool = True,
    output_size: Tuple[int, int] = (128, 128),
) -> torch.Tensor:


    
    stft = np.abs(librosa.stft(
        waveform,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window="hann",
    )) ** 2   

    fb = _linear_filterbank(sr, n_fft, n_filter)  
    filtered = np.dot(fb, stft)                    
    filtered = np.log(filtered + 1e-8)

    
    from scipy.fftpack import dct  
    lfcc = dct(filtered, type=2, axis=0, norm="ortho")[:n_lfcc, :]  

    if with_delta:
        delta  = librosa.feature.delta(lfcc, order=1)
        delta2 = librosa.feature.delta(lfcc, order=2)
        lfcc   = np.concatenate([lfcc, delta, delta2], axis=0)      

    
    lfcc = (lfcc - lfcc.mean(axis=1, keepdims=True)) / (lfcc.std(axis=1, keepdims=True) + 1e-8)
    lfcc = lfcc.astype(np.float32)
    return _resize(lfcc, output_size)


def extract_cqt(
    waveform: np.ndarray,
    sr: int = 16_000,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    hop_length: int = 256,
    fmin: Optional[float] = None,
    output_size: Tuple[int, int] = (128, 128),
) -> torch.Tensor:


    if fmin is None:
        fmin = librosa.note_to_hz("C1")   

    C = np.abs(librosa.cqt(
        waveform,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    ))   

    
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    C_norm = (C_db - C_db.min()) / (C_db.max() - C_db.min() + 1e-8)
    return _resize(C_norm.astype(np.float32), output_size)


def spec_augment(
    feature: torch.Tensor,
    freq_mask_param: int = 27,
    time_mask_param: int = 40,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:


    feat = feature.clone()
    _, H, W = feat.shape

    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, H - f))
        feat[:, f0 : f0 + f, :] = 0.0

    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(0, W - t))
        feat[:, :, t0 : t0 + t] = 0.0

    return feat


class FeatureExtractor:


    def __init__(self, cfg, augment: bool = False) -> None:
        self.cfg = cfg
        self.augment = augment

        fc = cfg.features
        self.sr           = cfg.data.sample_rate
        self.max_duration = cfg.data.max_duration
        self.output_size  = tuple(fc.output_size)

        
        m = fc.mel
        self.mel_kw = dict(
            sr=self.sr, n_mels=m.n_mels, n_fft=m.n_fft,
            win_length=m.win_length, hop_length=m.hop_length,
            fmax=m.fmax, power=m.power, top_db=m.top_db,
            output_size=self.output_size,
        )

        
        lf = fc.lfcc
        self.lfcc_kw = dict(
            sr=self.sr, n_filter=lf.n_filter, n_lfcc=lf.n_lfcc,
            n_fft=lf.n_fft, win_length=lf.win_length,
            hop_length=lf.hop_length, with_delta=lf.with_delta,
            output_size=self.output_size,
        )

        
        cq = fc.cqt
        self.cqt_kw = dict(
            sr=self.sr, n_bins=cq.n_bins,
            bins_per_octave=cq.bins_per_octave,
            hop_length=cq.hop_length, fmin=cq.fmin,
            output_size=self.output_size,
        )

        
        aug = fc.augment
        self.aug_kw = dict(
            freq_mask_param=aug.freq_mask_param,
            time_mask_param=aug.time_mask_param,
            num_freq_masks=aug.num_freq_masks,
            num_time_masks=aug.num_time_masks,
        )

    def __call__(
        self, waveform: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


        wav = _fix_length(waveform, self.sr, self.max_duration)

        mel  = extract_mel(wav,  **self.mel_kw)
        lfcc = extract_lfcc(wav, **self.lfcc_kw)
        cqt  = extract_cqt(wav,  **self.cqt_kw)

        if self.augment and self.cfg.features.augment.enabled:
            mel  = spec_augment(mel,  **self.aug_kw)
            lfcc = spec_augment(lfcc, **self.aug_kw)
            cqt  = spec_augment(cqt,  **self.aug_kw)

        return mel, lfcc, cqt

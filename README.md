<div align="center">

# 🎙️ MFCAN
### Multi-Feature Cross-Attention Network for Audio Deepfake Detection

[![Dataset](https://img.shields.io/badge/Dataset-ASVspoof%202019%20LA-blueviolet?style=for-the-badge&logo=databricks&logoColor=white)](https://www.asvspoof.org/)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/github/license/dbhaskarganesh/repo?style=for-the-badge&color=green)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/dbhaskarganesh/repo?style=for-the-badge&logo=github&color=yellow)](https://github.com/dbhaskarganesh/repo/stargazers)

**MFCAN v2** is a multi-feature fusion model that combines Mel-Spectrogram, LFCC, and CQT features with a cross-attention mechanism to detect AI-generated (spoofed) audio from real (bonafide) speech.

[📊 Results](#-results) • [🛠️ Architecture](#️-architecture) • [🚀 Getting Started](#-getting-started) • [🧪 Ablation Study](#-ablation-study) • [🐛 Report Bug](https://github.com/dbhaskarganesh/repo/issues)

</div>

---

## 📋 Table of Contents

- [✨ Overview](#-overview)
- [🛠️ Architecture](#️-architecture)
- [📁 Project Structure](#-project-structure)
- [⚡ Getting Started](#-getting-started)
- [🏋️ Training](#️-training)
- [🧪 Evaluation](#-evaluation)
- [📊 Results](#-results)
- [🔬 Ablation Study](#-ablation-study)
- [⚙️ Configuration](#️-configuration)
- [📄 License](#-license)
- [📬 Contact](#-contact)

---

## ✨ Overview

Audio deepfakes pose a growing threat to voice-based authentication systems. **MFCAN** addresses this by fusing three complementary acoustic features through a cross-attention mechanism, enabling the model to detect subtle artifacts introduced by TTS and voice conversion systems.

| 🎯 Task | 📦 Dataset | 🏷️ Classes | 📐 Params |
|--------|-----------|-----------|----------|
| Audio Deepfake Detection | ASVspoof 2019 LA | Spoof / Bonafide | 8.24M |

---

## 🛠️ Architecture

```
Input Audio
    │
    ├──▶ 🎵 Mel-Spectrogram
    ├──▶ 🔊 LFCC (60 + Δ + ΔΔ)
    └──▶ 🎼 CQT (84 bins)
              │
              ▼
    ┌─────────────────────┐
    │  Cross-Attention     │  8 heads × 2 layers
    │  (embed_dim = 256)   │  seq_len = 16
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  SE Block            │  reduction = 16
    └─────────────────────┘
              │
              ▼
    🟢 Bonafide  /  🔴 Spoof
```

**Key design choices:**
- 🔀 **Multi-feature fusion** — Mel, LFCC, and CQT capture different spectral/temporal characteristics
- 🔗 **Cross-attention** — 8-head, 2-layer attention models inter-feature dependencies
- 🎛️ **SE Block** — Squeeze-and-Excitation for channel-wise recalibration
- ⚖️ **Focal Loss** — γ=2.0 to handle class imbalance (89.7% spoof in eval set)

---

## 📁 Project Structure

```
📦 mfcan
 ┣ 📂 data               # 📥 ASVspoof 2019 LA dataset
 ┣ 📂 models             # 🧠 MFCAN architecture (mfcan.py)
 ┣ 📂 utils              # 🔧 Feature extraction, visualize.py
 ┣ 📂 Results            # 📊 Metrics, plots, confusion matrices
 ┣ 📜 train.py           # 🏋️ Training pipeline
 ┣ 📜 evaluate.py        # 🧪 Evaluation & threshold calibration
 ┣ 📜 config.yaml        # ⚙️ Hyperparameters & paths
 ┣ 📜 __init__.py        # 🚪 Package initializer
 ┣ 📜 LICENSE
 ┗ 📜 README.md
```

---

## ⚡ Getting Started

### ✅ Prerequisites

```bash
Python >= 3.8
CUDA (optional but recommended)
```

### 🔧 Installation

```bash
# 📥 Clone the repository
git clone https://github.com/dbhaskarganesh/mfcan.git
cd mfcan

# 📦 Install dependencies
pip install -r requirements.txt
```

### 📂 Prepare Data

Download the [ASVspoof 2019 LA partition](https://www.asvspoof.org/index2019.html) and place it as:

```
📂 data
 ┣ 📂 ASVspoof2019_LA_train
 ┣ 📂 ASVspoof2019_LA_dev
 ┗ 📂 ASVspoof2019_LA_eval
```

---

## ⚙️ Configuration

```yaml
# config.yaml
model:
  architecture: MFCAN v2
  embed_dim: 256
  cross_attention_heads: 8
  cross_attention_layers: 2
  attention_seq_len: 16
  se_reduction: 16

training:
  epochs: 50
  batch_size: 32
  optimizer: AdamW
  learning_rate: 0.0001
  weight_decay: 0.0001
  scheduler: CosineAnnealingWarmRestarts
  early_stopping_patience: 10
  loss: focal
  focal_gamma: 2.0
  inconsistency_loss_weight: 0.1
```

---

## 🏋️ Training

```bash
# 🚀 Train MFCAN
python train.py --config config.yaml
```

- Best checkpoint saved to `./checkpoints/mfcan_best.pt`
- Early stopping triggers after **10** epochs of no improvement
- Model converged at **epoch 12** / 50

---

## 🧪 Evaluation

```bash
# 🔍 Evaluate on ASVspoof 2019 LA eval set
python evaluate.py --checkpoint ./checkpoints/mfcan_best.pt
```

> Threshold is re-calibrated on the dev set via **tDCF minimisation** (`calibrated_threshold = -0.3503`)

---

## 📊 Results

Evaluated on **ASVspoof 2019 LA** eval set (71,237 samples — 89.68% spoof).

### 🏆 Primary Metrics

| Metric | Score |
|--------|-------|
| 🎯 **EER (%)** | **8.59** |
| 📉 **min-tDCF** | 0.9982 |
| 📈 **AUC-ROC** | 0.9698 |
| ✅ **Accuracy** | **73.93%** |

### 🔍 Classification Metrics

| Metric | Macro | Weighted |
|--------|-------|----------|
| 🎯 **Accuracy** | — | **73.93%** |
| 🏅 **F1 Score** | **0.6355** | **0.7899** |
| 🔬 **Precision** | **0.6413** | **0.9255** |
| 📡 **Recall** | **0.8530** | **0.7393** |

### 📂 Per-Class Breakdown

| Class | Precision | Recall | F1 | Samples |
|-------|-----------|--------|----|---------|
| 🔴 Spoof | 0.9994 | 0.7097 | 0.8300 | 63,882 |
| 🟢 Bonafide | 0.2832 | 0.9963 | 0.4411 | 7,355 |

### 🗃️ Confusion Matrix

```
                Predicted Spoof    Predicted Bonafide
Actual Spoof        45,338              18,544
Actual Bonafide         27               7,328
```

---

## 🔬 Ablation Study

Comparing feature combinations on ASVspoof 2019 LA eval set:

| Configuration | EER (%) ↓ | AUC-ROC ↑ | Accuracy ↑ | F1 Macro ↑ |
|---------------|-----------|-----------|-----------|------------|
| Only Mel | 13.46 | 0.9372 | 68.53% | 0.5909 |
| Only LFCC | 10.20 | 0.9607 | 85.83% | 0.7502 |
| Only CQT | 10.35 | 0.9624 | 91.78% | 0.8162 |
| Mel + LFCC | 8.03 | 0.9748 | 77.80% | 0.6695 |
| **Mel + CQT** ⭐ | **7.48** | **0.9739** | **86.02%** | **0.7527** |
| LFCC + CQT | 7.90 | 0.9750 | 89.50% | 0.7961 |
| No Cross-Attention | 9.78 | 0.9623 | 92.34% | 0.8230 |
| **Full MFCAN** | 8.59 | 0.9698 | 73.93% | 0.6355 |

> ⭐ Mel + CQT achieves the best EER (7.48%) — cross-attention adds robustness at the cost of some accuracy due to class imbalance sensitivity.

---

## 📄 License

Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for more information.

---

## 📬 Contact

👤 **dbhaskarganesh**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/dbhaskarganesh)

---

<div align="center">

⭐ **If this project helped you, please consider giving it a star!** ⭐

Made with ❤️ by [dbhaskarganesh](https://github.com/dbhaskarganesh)

</div>

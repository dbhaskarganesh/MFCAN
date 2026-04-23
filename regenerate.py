# regenerate_heatmap.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from omegaconf import OmegaConf
from models.mfcan import build_model
from data.dataset import build_datasets
from torch.utils.data import DataLoader
from utils.visualize import plot_attention_heatmap

cfg   = OmegaConf.load("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg).to(device)
ckpt  = torch.load("./checkpoints/mfcan_best.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

_, _, eval_ds = build_datasets(cfg)
loader = DataLoader(eval_ds, batch_size=4, shuffle=False)
batch  = next(iter(loader))

with torch.no_grad():
    model(batch["mel"].to(device), batch["lfcc"].to(device), batch["cqt"].to(device))

attn = model.get_attention_weights()
plot_attention_heatmap(attn, save_path="./results/attention_heatmap.png")
print("Saved.")
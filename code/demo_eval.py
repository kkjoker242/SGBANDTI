# -*- coding: utf-8 -*-
"""加载样本模型并对 BioSNAP random 测试集评估（不重训）。
用法：
  python demo_eval.py                                # 用 models/ 下的样本模型
  python demo_eval.py --checkpoint path/to/model.pth # 自定义权重
预期输出（样本模型 seed42）≈ AUROC 0.9062（与论文一致）。
"""
import argparse
import os
import re
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from configs import get_cfg_defaults
from dataloader import DTIDataset, collate_fn_nested
from models import SGBANDTI
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Evaluate sample SGBANDTI model")
parser.add_argument("--data", default="biosnap", choices=["bindingdb", "biosnap"])
parser.add_argument("--split", default="random")
parser.add_argument("--hop", type=int, default=2)
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = args.checkpoint or "../models/SGBANDTI_biosnap_random_seed42.pth"
if not os.path.exists(ckpt):
    raise FileNotFoundError(f"模型不存在: {ckpt}")

cfg = get_cfg_defaults()
cfg.SOLVER.SEED = 42
model = SGBANDTI(**cfg).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()

df = pd.read_csv(os.path.join("../data", args.data, args.split, "test.csv"))
ds = DTIDataset(df.index.values, df, dataset_name=args.data, split_name=args.split,
                split_file_name="test", h=args.hop, use_nested=True)
dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, drop_last=False,
                collate_fn=collate_fn_nested)

y_true, y_prob = [], []
with torch.no_grad():
    for bg, prot, labels in dl:
        bg, prot = bg.to(device), prot.to(device)
        _, _, score, _ = model(bg, prot, mode="eval")
        y_true += labels.tolist()
        y_prob += torch.squeeze(score, 1).cpu().tolist()

y_true = np.asarray(y_true)
y_prob = np.asarray(y_prob)
print(f"样本模型: {ckpt}")
print(f"测试样本数: {len(y_true)}")
print(f"AUROC: {roc_auc_score(y_true, y_prob):.4f}  (论文 seed42 ≈ 0.9062)")
print(f"AUPRC: {average_precision_score(y_true, y_prob):.4f}  (论文 seed42 ≈ 0.9132)")

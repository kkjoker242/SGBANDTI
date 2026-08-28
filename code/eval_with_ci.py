# -*- coding: utf-8 -*-
"""
评估已保存检查点，并输出 AUROC/AUPRC 的 paired bootstrap 置信区间（项11）。

示例：
  python eval_with_ci.py --data biosnap --split random --hop 2 --seed 42
  python eval_with_ci.py --checkpoint result/biosnap_random_hop2/seed_42/best_model_epoch_XX.pth
"""
import argparse
import os
import re
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from configs import get_cfg_defaults
from dataloader import DTIDataset, collate_fn_nested
from models import SGBANDTI
from utils import bootstrap_metric_ci, mkdir, set_seed
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Evaluate checkpoint with bootstrap CI")
parser.add_argument("--data", default="biosnap", choices=["bindingdb", "biosnap", "human"])
parser.add_argument("--split", default="random", choices=["random", "cold", "cluster", "unseen_drug", "unseen_target"])
parser.add_argument("--hop", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", default=None, help="path to best_model_epoch_*.pth")
parser.add_argument("--n-boot", type=int, default=1000)
parser.add_argument("--ci-seed", type=int, default=42)
parser.add_argument("--ablation", default="full",
                    choices=["full", "no_subgraph", "no_ban", "no_both"])
args = parser.parse_args()


def find_best_checkpoint():
    tag = "" if args.ablation == "full" else f"_{args.ablation}"
    d = os.path.join("./result", f"{args.data}_{args.split}_hop{args.hop}{tag}", f"seed_{args.seed}")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"checkpoint dir not found: {d}")
    cands = []
    for f in os.listdir(d):
        m = re.match(r"best_model_epoch_(\d+)\.(pth|pt)$", f)
        if m:
            cands.append((int(m.group(1)), os.path.join(d, f)))
    if not cands:
        raise FileNotFoundError(f"no best model in {d}")
    cands.sort()
    return cands[-1][1]


def main():
    torch.cuda.empty_cache()
    cfg = get_cfg_defaults()
    if args.ablation in ("no_subgraph", "no_both"):
        cfg.ABLATION.USE_SUBGRAPH = False
    if args.ablation in ("no_ban", "no_both"):
        cfg.ABLATION.USE_BAN = False
    cfg.SOLVER.SEED = args.seed
    set_seed(args.seed)
    ckpt = args.checkpoint or find_best_checkpoint()

    df = pd.read_csv(os.path.join("./datasets", args.data, args.split, "test.csv"))
    ds = DTIDataset(df.index.values, df, dataset_name=args.data, split_name=args.split,
                    split_file_name="test", h=args.hop, use_nested=cfg.ABLATION.USE_SUBGRAPH)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0,
                    drop_last=False, collate_fn=collate_fn_nested)

    model = SGBANDTI(**cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for bg, prot, labels in dl:
            bg, prot = bg.to(device), prot.to(device)
            v_d, v_p, score, att = model(bg, prot, mode="eval")
            y_true += labels.tolist()
            y_pred += torch.squeeze(score, 1).cpu().tolist()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 保存逐样本预测概率（P0-#7），供指标重算与配对 bootstrap 使用
    save_dir = os.path.dirname(ckpt)
    np.save(os.path.join(save_dir, "test_y_true.npy"), y_true)
    np.save(os.path.join(save_dir, "test_y_pred.npy"), y_pred)
    print(f"逐样本预测已保存: {save_dir}/test_y_true.npy, test_y_pred.npy")

    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    auc_m, auc_lo, auc_hi = bootstrap_metric_ci(y_true, y_pred, "auroc", args.n_boot, args.ci_seed)
    ap_m, ap_lo, ap_hi = bootstrap_metric_ci(y_true, y_pred, "auprc", args.n_boot, args.ci_seed)

    print(f"checkpoint: {ckpt}")
    print(f"样本数: {len(y_true)}")
    print(f"AUROC: {auroc:.4f}  (bootstrap 95% CI [{auc_lo:.4f}, {auc_hi:.4f}], mean {auc_m:.4f})")
    print(f"AUPRC: {auprc:.4f}  (bootstrap 95% CI [{ap_lo:.4f}, {ap_hi:.4f}], mean {ap_m:.4f})")


if __name__ == "__main__":
    main()

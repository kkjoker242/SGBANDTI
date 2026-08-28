# -*- coding: utf-8 -*-
"""
从 result/<data>_<split>_hop<hop>[<ablation>]/seed_*/result_metrics.pt
聚合所有已完成 seed 的测试指标，生成 seed_summary.csv 与 seed_summary_stats.csv。

用法：
  python aggregate_seeds.py --data biosnap --split random --hop 2
  python aggregate_seeds.py --data biosnap --split random --hop 2 --ablation no_ban
"""
import argparse
import glob
import os

import pandas as pd
import torch

parser = argparse.ArgumentParser(description="Aggregate per-seed test metrics")
parser.add_argument("--data", default="biosnap")
parser.add_argument("--split", default="random")
parser.add_argument("--hop", type=int, default=2)
parser.add_argument("--ablation", default="full",
                    choices=["full", "no_subgraph", "no_ban", "no_both"])
args = parser.parse_args()

tag = "" if args.ablation == "full" else f"_{args.ablation}"
base = os.path.join("result", f"{args.data}_{args.split}_hop{args.hop}{tag}")

KEYS = ["auroc", "auprc", "test_loss", "sensitivity", "specificity",
        "accuracy", "thred_optim", "best_epoch", "F1", "Precision"]

rows = []
for d in sorted(glob.glob(os.path.join(base, "seed_*"))):
    mp = os.path.join(d, "result_metrics.pt")
    if not os.path.isfile(mp):
        print(f"skip {d}（无 result_metrics.pt，未完成）")
        continue
    st = torch.load(mp, map_location="cpu")
    m = st["test_metrics"]
    seed = int(os.path.basename(d).split("_")[1])
    rows.append({"seed": seed, **{k: m.get(k) for k in KEYS}})

if not rows:
    print(f"未找到任何已完成的 seed 结果：{base}")
    raise SystemExit(1)

summary = pd.DataFrame(rows).sort_values("seed")[["seed"] + KEYS]
summary.to_csv(os.path.join(base, "seed_summary.csv"), index=False)
stats = summary[KEYS].agg(["mean", "std"])
stats.to_csv(os.path.join(base, "seed_summary_stats.csv"))

print(f"已聚合 {len(rows)} 个 seed：{sorted(summary['seed'].tolist())}")
print(summary.to_string(index=False))
print("\nmean ± std：")
for c in ["auroc", "auprc", "F1", "sensitivity", "specificity", "accuracy"]:
    print(f"  {c:12s}: {stats[c]['mean']:.4f} ± {stats[c]['std']:.4f}")
print(f"\n已写入 {base}/seed_summary.csv 与 seed_summary_stats.csv")

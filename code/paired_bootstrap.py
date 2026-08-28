# -*- coding: utf-8 -*-
"""
配对 bootstrap（P0-#9）：比较 SGBANDTI 与某基线的 ΔAUROC/ΔAUPRC 及 95% CI。

输入为两个模型在同一测试集上的逐样本预测概率（由 eval_with_ci.py 保存的
test_y_true.npy / test_y_pred.npy）。配对 bootstrap：同一次重采样索引同时用于
两个模型，得到 Δ 的分布，若 95% CI 不含 0 则差异显著。

用法：
  python paired_bootstrap.py \
    --y-true 路径/y_true.npy \
    --a 路径A/test_y_pred.npy --name-a SGBANDTI \
    --b 路径B/test_y_pred.npy --name-b MGNDTI \
    --n-boot 1000 --seed 42
"""
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser()
parser.add_argument("--y-true", required=True, help="共享测试集标签 .npy")
parser.add_argument("--a", required=True, help="模型A 预测概率 .npy")
parser.add_argument("--name-a", default="A")
parser.add_argument("--b", required=True, help="模型B 预测概率 .npy")
parser.add_argument("--name-b", default="B")
parser.add_argument("--n-boot", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


def paired_delta(y_true, pred_a, pred_b, metric, n_boot, seed):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    fn = roc_auc_score if metric == "auroc" else average_precision_score
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            da = fn(y_true[idx], pred_a[idx])
            db = fn(y_true[idx], pred_b[idx])
            diffs.append(da - db)
        except ValueError:
            continue
    diffs = np.asarray(diffs)
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def main():
    y_true = np.load(args.y_true)
    pa = np.load(args.a)
    pb = np.load(args.b)
    assert len(pa) == len(pb) == len(y_true), "样本数不一致！"

    for metric in ("auroc", "auprc"):
        va, vb = (roc_auc_score if metric == "auroc" else average_precision_score)(y_true, pa), \
                 (roc_auc_score if metric == "auroc" else average_precision_score)(y_true, pb)
        mean, lo, hi = paired_delta(y_true, pa, pb, metric, args.n_boot, args.seed)
        sig = "✅ 显著" if (lo > 0 or hi < 0) else "不显著"
        print(f"{metric.upper():5s}: {args.name_a}={va:.4f}  {args.name_b}={vb:.4f}  "
              f"Δ={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
统一评价脚本（P0-#8）：用标准定义重算六项指标，并校验指标恒等关系。

修复主表数学矛盾：Accuracy = π·Sensitivity + (1-π)·Specificity 必须成立
（π 为固定测试集阳性率）。若同一测试集下各方法反推的 π 不一致，说明混用了
不同划分/评价协议。

用法：
  python eval_metrics.py --y-true y_true.npy --y-prob y_prob.npy --threshold 0.5
  或作为库：from eval_metrics import compute_metrics
"""
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def compute_metrics(y_true, y_prob, threshold):
    """按统一标准计算六项指标并校验恒等关系。

    Args:
        y_true: 二分类标签 (n,)
        y_prob: 预测概率 (n,)
        threshold: 分类阈值（应为验证集选出的固定值）
    Returns:
        dict: auroc, auprc, f1, sensitivity, specificity, accuracy, implied_pi, identity_ok, n
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    pi = float(y_true.mean())  # 测试集阳性率

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    accuracy = (tp + tn) / n if n > 0 else float("nan")
    f1 = f1_score(y_true, y_pred)

    # 恒等校验：由 Accuracy/Sens/Spec 反推 π，应与真实 π 一致
    if (sensitivity - specificity) != 0 and not np.isnan(sensitivity):
        implied_pi = (accuracy - specificity) / (sensitivity - specificity)
    else:
        implied_pi = float("nan")
    identity_ok = abs(implied_pi - pi) < 1e-3

    return {
        "n": n, "pi": pi, "threshold": threshold,
        "auroc": auroc, "auprc": auprc,
        "f1": f1, "sensitivity": sensitivity,
        "specificity": specificity, "accuracy": accuracy,
        "implied_pi": implied_pi, "identity_ok": identity_ok,
    }


def print_metrics(m):
    print(f"样本数 n={m['n']} | 阳性率 π={m['pi']:.4f} | 阈值={m['threshold']:.4f}")
    print(f"  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}")
    print(f"  F1={m['f1']:.4f}  Sensitivity={m['sensitivity']:.4f}  "
          f"Specificity={m['specificity']:.4f}  Accuracy={m['accuracy']:.4f}")
    print(f"  反推 π={m['implied_pi']:.4f} | 恒等校验(Accuracy=π·Sens+(1-π)·Spec): "
          f"{'✅ 通过' if m['identity_ok'] else '❌ 不成立（指标冲突）'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--y-true", required=True, help=".npy 标签文件")
    p.add_argument("--y-prob", required=True, help=".npy 预测概率文件")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    y_true = np.load(args.y_true)
    y_prob = np.load(args.y_prob)
    m = compute_metrics(y_true, y_prob, args.threshold)
    print_metrics(m)
    if not m["identity_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

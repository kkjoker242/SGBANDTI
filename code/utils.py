import os
import random
import numpy as np
import torch
import logging

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def bootstrap_metric_ci(y_true, y_pred, metric="auroc", n_boot=1000, seed=42):
    """对二分类指标做 paired bootstrap 置信区间（百分位法，项11）。

    Args:
        y_true, y_pred: 标签与预测分数（等长）。
        metric: "auroc" 或 "auprc"。
        n_boot: 重采样次数。
        seed: 随机种子。
    Returns:
        (mean, ci_low, ci_high)
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    metric_fn = roc_auc_score if metric == "auroc" else average_precision_score
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except ValueError:
            # 单类样本的 bootstrap 重采样可能退化，跳过
            continue
    scores = np.asarray(scores)
    if len(scores) == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(scores.mean()),
        float(np.percentile(scores, 2.5)),
        float(np.percentile(scores, 97.5)),
    )


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding

"""解析各调参跑的 valid_markdowntable.txt, 提取 best val AUROC/AUPRC, 写 tune_report.md。
只在验证集上比较, 绝不看 test 挑配置。"""
import os
import re
from datetime import datetime


def best_val(seed_dir):
    path = os.path.join(seed_dir, "valid_markdowntable.txt")
    if not os.path.exists(path):
        return None
    best = (0.0, 0.0, 0)  # auroc, auprc, epoch
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r"\|\s*epoch\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
            if m:
                ep, auroc, auprc = int(m.group(1)), float(m.group(2)), float(m.group(3))
                if auroc > best[0]:
                    best = (auroc, auprc, ep)
    return best


ROOT = "result"
RUNS = [
    ("existing_final_baseline", os.path.join(ROOT, "biosnap_random_hop2", "seed_42"),
     "最终协议 baseline（现有, test AUROC 0.9062）"),
    ("tune_baseline", os.path.join(ROOT, "biosnap_random_hop2_tune_baseline", "seed_42"),
     "回归 baseline（默认参数）"),
    ("tune_dropout02_wd1e4", os.path.join(ROOT, "biosnap_random_hop2_tune_dropout02_wd1e4", "seed_42"),
     "批次1 dropout0.2 + weight_decay 1e-4"),
    ("tune_heads4", os.path.join(ROOT, "biosnap_random_hop2_tune_heads4", "seed_42"),
     "批次2 BAN heads=4"),
    ("tune_lr1e4", os.path.join(ROOT, "biosnap_random_hop2_tune_lr1e4", "seed_42"),
     "批次4 lr=1e-4"),
    ("tune_hidden256", os.path.join(ROOT, "biosnap_random_hop2_tune_hidden256", "seed_42"),
     "批次4 hidden=256 (drug+protein 128→256)"),
    ("tune_bs32", os.path.join(ROOT, "biosnap_random_hop2_tune_bs32", "seed_42"),
     "批次5 batch_size=32"),
    ("tune_cosine", os.path.join(ROOT, "biosnap_random_hop2_tune_cosine", "seed_42"),
     "批次5 lr1e-4 + cosine"),
    ("tune_maxepoch250", os.path.join(ROOT, "biosnap_random_hop2_tune_maxepoch250", "seed_42"),
     "批次5 MAX_EPOCH=250"),
]

lines = ["# 超参微调自动链报告", "",
         f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
         "比较指标 = **验证集** best val AUROC（红线: 绝不看 test 挑配置）。", "",
         "| 配置 | best val AUROC | best val AUPRC | best epoch | 状态 |",
         "|---|---|---|---|---|"]
for name, path, desc in RUNS:
    res = best_val(path)
    if res is None:
        lines.append(f"| {desc} (`{name}`) | — | — | — | **缺失/失败** |")
    else:
        auroc, auprc, ep = res
        lines.append(f"| {desc} (`{name}`) | {auroc:.4f} | {auprc:.4f} | {ep} | OK |")


def judge(label, res):
    if res is None:
        return f"- **{label}**: 跑失败/结果缺失, 无法判定。"
    if base is None:
        return f"- **{label}**: 无基线可对比。"
    d = res[0] - base[0]
    verdict = "胜出" if d > 0.001 else ("打平" if abs(d) <= 0.001 else "更差")
    return f"- **{label}**: val AUROC {res[0]:.4f} vs 现有基线 {base[0]:.4f}（Δ={d:+.4f}）→ **{verdict}**"


base = best_val(RUNS[0][1])
res_base = best_val(RUNS[1][1])
res_v1 = best_val(RUNS[2][1])
res_v2 = best_val(RUNS[3][1])
res_v3 = best_val(RUNS[4][1])
res_v4 = best_val(RUNS[5][1])
res_v5 = best_val(RUNS[6][1])
res_v6 = best_val(RUNS[7][1])
res_v7 = best_val(RUNS[8][1])

lines += ["", "## 判定", ""]
lines.append(f"- 基线参考: 现有最终协议 val AUROC {base[0]:.4f}（epoch {base[2]}）" if base else "- 基线缺失!")
if res_base:
    d = res_base[0] - base[0]
    lines.append(f"- 回归 baseline（默认参数）: {res_base[0]:.4f}（Δ={d:+.4f}）→ 确认代码改动是否可复现。")
lines.append(judge("批次1 dropout+wd", res_v1))
lines.append(judge("批次2 heads4", res_v2))
lines.append(judge("批次4 lr=1e-4", res_v3))
lines.append(judge("批次4 hidden=256", res_v4))
lines.append(judge("批次5 batch_size=32", res_v5))
lines.append(judge("批次5 lr1e-4 + cosine", res_v6))
lines.append(judge("批次5 MAX_EPOCH=250", res_v7))
lines += ["", "## 建议", "",
          "- 单种子筛选有噪声（±0.001-0.002），明显更差的配置直接放弃；打平或胜出的进候选。",
          "- 候选 → 人工决定是否进批次3（5种子确认 ~13h, 反超 DrugBAN 0.9100 才定版）。",
          "- 若多个变体胜出, 可考虑合成后再单种子确认一次。"]

report = "\n".join(lines) + "\n"
with open("tune_report.md", "w", encoding="utf-8") as f:
    f.write(report)
print(report)

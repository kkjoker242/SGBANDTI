# 超参微调自动链报告

生成时间: 2026-08-20 05:50:00

比较指标 = **验证集** best val AUROC（红线: 绝不看 test 挑配置）。

| 配置 | best val AUROC | best val AUPRC | best epoch | 状态 |
|---|---|---|---|---|
| 最终协议 baseline（现有, test AUROC 0.9062） (`existing_final_baseline`) | 0.9061 | 0.9107 | 137 | OK |
| 回归 baseline（默认参数） (`tune_baseline`) | 0.9027 | 0.9094 | 141 | OK |
| 批次1 dropout0.2 + weight_decay 1e-4 (`tune_dropout02_wd1e4`) | 0.8906 | 0.8910 | 143 | OK |
| 批次2 BAN heads=4 (`tune_heads4`) | 0.9063 | 0.9063 | 132 | OK |
| 批次4 lr=1e-4 (`tune_lr1e4`) | 0.9043 | 0.9124 | 121 | OK |
| 批次4 hidden=256 (drug+protein 128→256) (`tune_hidden256`) | 0.9028 | 0.9065 | 127 | OK |
| 批次5 batch_size=32 (`tune_bs32`) | 0.9062 | 0.9118 | 150 | OK |
| 批次5 lr1e-4 + cosine (`tune_cosine`) | 0.9027 | 0.9109 | 148 | OK |
| 批次5 MAX_EPOCH=250 (`tune_maxepoch250`) | 0.9132 | 0.9213 | 231 | OK |

## 判定

- 基线参考: 现有最终协议 val AUROC 0.9061（epoch 137）
- 回归 baseline（默认参数）: 0.9027（Δ=-0.0034）→ 确认代码改动是否可复现。
- **批次1 dropout+wd**: val AUROC 0.8906 vs 现有基线 0.9061（Δ=-0.0155）→ **更差**
- **批次2 heads4**: val AUROC 0.9063 vs 现有基线 0.9061（Δ=+0.0002）→ **打平**
- **批次4 lr=1e-4**: val AUROC 0.9043 vs 现有基线 0.9061（Δ=-0.0018）→ **更差**
- **批次4 hidden=256**: val AUROC 0.9028 vs 现有基线 0.9061（Δ=-0.0033）→ **更差**
- **批次5 batch_size=32**: val AUROC 0.9062 vs 现有基线 0.9061（Δ=+0.0001）→ **打平**
- **批次5 lr1e-4 + cosine**: val AUROC 0.9027 vs 现有基线 0.9061（Δ=-0.0034）→ **更差**
- **批次5 MAX_EPOCH=250**: val AUROC 0.9132 vs 现有基线 0.9061（Δ=+0.0071）→ **胜出**

## 建议

- 单种子筛选有噪声（±0.001-0.002），明显更差的配置直接放弃；打平或胜出的进候选。
- 候选 → 人工决定是否进批次3（5种子确认 ~13h, 反超 DrugBAN 0.9100 才定版）。
- 若多个变体胜出, 可考虑合成后再单种子确认一次。

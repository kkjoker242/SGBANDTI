# SGBANDTI

**Version**: v1.0.0

SGBANDTI: **S**ubgraph-**G**NN and **B**ilinear **A**ttention for **D**rug-**T**arget **I**nteraction prediction.

SGBANDTI predicts drug–target interactions (DTI) by coupling an atom-centered **k-hop rooted subgraph** GNN drug encoder with a **low-rank bilinear attention (BAN)** protein fusion module, followed by an MLP classifier. This repository provides the complete code, data splits, and per-seed results used in the paper.

---

## Repository Structure

```
SGBANDTI/
├── README.md                 # this file
├── code/                     # SGBANDTI 核心代码 + 复现脚本
│   ├── main.py               # 训练/评估入口
│   ├── models.py             # SGBANDTI 模型
│   ├── gcn.py                # (Nested)GCN 编码器
│   ├── ban.py                # BAN 双线性注意力
│   ├── trainer.py            # 训练器（验证集选优 + 阈值持久化）
│   ├── configs.py            # 配置
│   ├── dataloader.py         # 数据加载 + 子图缓存
│   ├── build_subgraph_cache.py
│   ├── build_splits.py       # canonical 划分（防泄漏）
│   ├── eval_metrics.py       # 统一评价脚本
│   ├── eval_with_ci.py       # bootstrap CI + 逐样本导出
│   ├── paired_bootstrap.py   # 配对 bootstrap（vs 基线）
│   └── aggregate_seeds.py    # 5 种子聚合
├── baselines/                # 对比基线代码（7 个，见 README 基线节）
│   ├── DrugBAN/  MolTrans/  MGNDTI/  INGNN/
│   ├── TransformerCPI/  GNN-CPI/  RF/
├── data/                     # 固定 train/val/test 划分（7:1:2，canonical 防泄漏）
│   ├── biosnap/{random, unseen_drug, unseen_target}
│   └── bindingdb/random
└── results/                  # 全部实验结果（单独存放）
    ├── 00_实验结果汇总.md     # 论文表格数据（mean±std + 逐种子）
    ├── 01_调参过程与结果.md   # 超参微调记录
    ├── per_seed/             # 逐种子原始结果文件（result_metrics.pt / 表格 / 图）
    └── tables/               # 逐种子明细表格 + CSV
```

> 代码、数据、基线、文档与**结果分离**：`results/` 独立存放，便于直接引用与核对。

---

## Requirements

**环境**：Python 3.10.20 · CUDA 12.1（cu121）· NVIDIA GPU（RTX 4060 / 4090 实测）· Linux / Windows 均可。

核心依赖：

| 包 | 版本 |
|---|---|
| torch | 2.1.0+cu121 |
| dgl | 2.2.1+cu121 |
| torch-scatter | 2.1.2+pt21cu121 |
| torchdata | **0.7.1** |
| dgllife | 0.3.2 |
| torch-geometric | 2.6.1 |
| rdkit | 2026.3.5 |
| numpy / pandas / scikit-learn / yacs / prettytable / tqdm | 标准 |

```bash
pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install dgl==2.2.1+cu121 -f https://data.dgl.ai/wheels/cu121/repo.html
pip install torch-scatter==2.1.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torchdata==0.7.1
```

---

## Data

`data/` 提供去重后（BioSNAP 移除 7 个重复对）的固定划分：
- **BioSNAP**：random / unseen_drug（药物冷启动）/ unseen_target（蛋白冷启动），train 19,220+ / val 2,746 / test 5,491（random）。
- **BindingDB**：random，train 34,439 / val 4,920 / test 9,840。
- 冷启动按 canonical SMILES / 标准化蛋白分组，train/val/test 实体交集 = 0（无泄漏）。
- **划分冻结**：各 split 的样本数/实体统计/md5 哈希见 `data/SPLITS_FROZEN.md`（`data/SPLITS_FROZEN.json` 为机器可读版）；任何数据改动会导致哈希变化。

**负样本构造**：
- **BioSNAP**：正样本 = 已知药物–蛋白相互作用；负样本 = 未标注的随机药物–蛋白对（接近平衡，正 13,830 / 负 13,627）。
- **BindingDB**：正样本 = 实验测得的结合记录；负样本 = 随机生成的未结合对（正 20,674 / 负 28,525，标签列为 Y：1 结合 / 0 未结合）。

首次运行需重建子图缓存：

```bash
python build_subgraph_cache.py --data biosnap --split random --hop 2 --use-nested nested
python build_subgraph_cache.py --data biosnap --split random --hop 2 --use-nested flat
python build_subgraph_cache.py --data biosnap --split unseen_drug --hop 2 --use-nested nested
python build_subgraph_cache.py --data biosnap --split unseen_target --hop 2 --use-nested nested
python build_subgraph_cache.py --data bindingdb --split random --hop 2 --use-nested nested
```

---

## Usage

训练（5 种子 42–82，模型按**验证集 AUROC** 选最优）：

```bash
# BioSNAP random 主实验（定版：AUROC 0.9062±0.0019）
python main.py --data biosnap --split random --hop 2 --seeds 42,52,62,72,82

# 消融 2×2
python main.py --data biosnap --split random --hop 2 --seeds 42,52,62,72,82 --ablation no_subgraph
python main.py --data biosnap --split random --hop 2 --seeds 42,52,62,72,82 --ablation no_ban
python main.py --data biosnap --split random --hop 2 --seeds 42,52,62,72,82 --ablation no_both

# 冷启动
python main.py --data biosnap --split unseen_drug --hop 2 --seeds 42,52,62,72,82
python main.py --data biosnap --split unseen_target --hop 2 --seeds 42,52,62,72,82

# BindingDB random
python main.py --data bindingdb --split random --hop 2 --seeds 42,52,62,72,82
```

统计与评价：

```bash
python aggregate_seeds.py --data biosnap --split random --hop 2        # 5 种子聚合
python eval_with_ci.py --data biosnap --split random --hop 2 --seed 42 # bootstrap CI + 逐样本
python paired_bootstrap.py --y-true ... --a SGB.npy --b DrugBAN.npy    # 配对 bootstrap
python eval_metrics.py --y-true ... --y-prob ... --threshold 0.5       # 统一指标校验
```

## Pretrained Model（样本模型）

`models/` 提供**一个代表模型**（SGBANDTI BioSNAP random seed42，best epoch 137），无需重训即可复现论文数字（AUROC **0.9062** / AUPRC **0.9171**）：

```bash
python demo_eval.py        # 需先建好子图缓存（见 Data 节）
```

其余种子与实验（冷启动 / BindingDB / 消融 / 基线）权重较大且可在本地重训，故不随仓库分发；对应结果见 `results/`。

> 阈值依赖指标（F1/Sens/Spe/Acc）统一用**验证集 F1 最优阈值**；cudnn 确定性已开（结果可复现）。模型默认配置 = 论文定版（参数 1,070,342）。

---

## Baselines

`baselines/` 提供 7 个对比基线代码（DrugBAN / MolTrans / MGNDTI / INGNN / TransformerCPI / GNN-CPI / RF），均已适配同一去重数据、5 种子 42–82 与统一指标口径（标准 F1/阈值=验证集选优）。各基线来源与适配说明：

| 基线 | 官方仓库 | 适配 |
|---|---|---|
| DrugBAN | github.com/peizhenbai/DrugBAN | 指标修正（标准 F1/Sens/Spe）、`--seeds` 循环、冷启动 split |
| MolTrans | github.com/kc-chenlab/MolTrans | Sens/Spe 语义修正、`--task biosnap/bindingdb/冷启动` |
| MGNDTI | github.com/Nicole-cui/MGNDTI | `--data/--split/--seeds` 支持 |
| INGNN | github.com/CSUBioGroup/iNGNN-DTI | ESPF 词表、`--split` 冷启动、seeds 42–82 |
| TransformerCPI | github.com/lucian02/TransformerCPI | `--data` 切换、冷启动支持 |
| GNN-CPI | github.com/IBM/Interpretable-... (CPI) | seeds 42–82 |
| RF | 自实现（RDKit 特征 + 随机森林） | 统一阈值协议 |

> 各基线已在同一去重数据、统一种子与统一指标口径下复现（见 `results/`）。GraphBAN 为 transductive 口径（与 random 不可直接比），论文中作为参考，代码见其官方仓库（github.com/HamidHadipour/GraphBAN）。

---

## Results

完整结果见 `results/`（与代码分离存放）：
- **`00_实验结果汇总.md`**：论文全部表格数据（mean±std + 逐种子）。
- **`01_调参过程与结果.md`**：超参微调记录（dropout/heads/lr/hidden/bs/epoch/cosine）。
- **`per_seed/`**：各实验逐种子原始文件（result_metrics.pt / markdown 表 / 图）。
- **`tables/`**：逐种子明细（全模型）+ `SGBANDTI_per_seed.csv`。

### 关键结果（AUROC，mean±std）

**BioSNAP random（5 种子）**

| 模型 | AUROC | AUPRC |
|---|---|---|
| **SGBANDTI** | **0.9062±0.0019** | 0.9132±0.0043 |
| DrugBAN | 0.9100±0.0023 | 0.9172±0.0031 |
| MolTrans | 0.8867±0.0045 | 0.8927±0.0048 |
| MGNDTI | 0.8947±0.0019 | 0.8983±0.0042 |

**冷启动 unseen_drug**：SGBANDTI 0.8794±0.0019 为最优；**unseen_target**：RF 0.6979±0.0110 最优。

**配对 bootstrap（SGBANDTI vs DrugBAN）**：BioSNAP random / BindingDB / unseen_drug 差异均**不显著**（95% CI 含 0）；unseen_target DrugBAN 显著更优。

---

## Citation

论文信息待补充（录用后请更新 `CITATION.cff`）。

## License

This project is licensed under the [MIT License](LICENSE)。

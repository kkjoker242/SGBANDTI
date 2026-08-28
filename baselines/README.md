# Baselines（对比基线复现说明）

> 7 个基线均已适配：**同一去重 canonical 数据、5 种子 42–82、统一指标口径**（标准 F1、阈值=验证集选优）。
> 各基线目录内含其官方代码与文档；本文件说明如何在本仓库数据上复现。
> 数据源：`../data/`（BioSNAP random/unseen_drug/unseen_target + BindingDB random）。使用前将对应 CSV 放入各基线预期目录（见下）。

## 环境

均基于 `sgbandti` 环境（见 `environment.yml`），个别基线需额外依赖：

| 基线 | 额外依赖 |
|---|---|
| MolTrans | `pip install subword-nmt` |
| TransformerCPI | `pip install gensim` |
| 其余 | 无 |

## 各基线数据放置与运行

| 基线 | 数据放置（从 ../data 拷入） | 运行命令 |
|---|---|---|
| **DrugBAN** | `datasets/biosnap/random`、`datasets/bindingdb/random`（及 unseen 冷启动） | `python main.py --cfg configs/DrugBAN.yaml --data biosnap --split random --seeds 42,52,62,72,82` |
| **MolTrans** | `dataset/BIOSNAP/random`、`dataset/bindingdb/random` | `python train_bindingdb.py --task biosnap --seeds 42 52 62 72 82` |
| **MGNDTI** | `datasets/biosnap/{random,unseen_drug,unseen_target}`、`datasets/bindingdb/random` | `python main.py --data biosnap --split random --seeds 42,52,62,72,82` |
| **INGNN** | `datasets/biosnap/{random,unseen_drug,unseen_target}`（+ `ESPF/` 词表已内置） | `python main.py`（main.py 内设 dataFolder/seeds） |
| **TransformerCPI** | `dataset/biosnap/random`、`dataset/bindingdb/random` | `python main_glu.py --data biosnap --seeds 42,52,62,72,82` |
| **GNN-CPI** | `dataset/{biosnap,bindingdb}/random` | `python run_training.py biosnap 2 3 10 3 11 3 3 0.001 0.5 10 0.000001 60 <setting> 42,52,62,72,82` |
| **RF** | `dataset/biosnap/random`、`dataset/bindingdb/random` | `python dti_prediction_bio.py biosnap` |

> 注意：各基线数据路径为相对路径，需从其各自目录下运行；冷启动（unseen_drug/unseen_target）对应数据放对应目录即可。
> 各基线结果与逐种子明细见 `../results/`。GraphBAN（transductive，与 random 不可直接比）未随仓库分发，见其官方仓库 github.com/HamidHadipour/GraphBAN。

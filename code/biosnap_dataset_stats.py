"""BioSNAP 数据集统计统一：去重后各 split 的唯一药物/蛋白/正负样本数，对齐论文口径。"""
import os
import pandas as pd

ROOT = "datasets/biosnap"

print("=== BioSNAP 数据集统计 ===")
for split in ["random", "unseen_drug", "unseen_target"]:
    print(f"\n--- {split} ---")
    for part in ["train", "val", "test"]:
        p = os.path.join(ROOT, split, f"{part}.csv")
        df = pd.read_csv(p)
        n = len(df)
        pos = int(df["Y"].sum())
        drugs = df["SMILES"].nunique()
        prot = df["Protein"].nunique()
        print(f"  {part}: n={n} 正={pos} 负={n-pos} 唯一药物={drugs} 唯一蛋白={prot}")

# 去重后总量(random 三段)
print("\n=== random 合计 ===")
total = 0
for part in ["train", "val", "test"]:
    df = pd.read_csv(os.path.join(ROOT, "random", f"{part}.csv"))
    total += len(df)
all_df = pd.concat([pd.read_csv(os.path.join(ROOT, "random", f"{p}.csv")) for p in ["train", "val", "test"]])
print(f"  random 总样本: {total}")
print(f"  唯一药物: {all_df['SMILES'].nunique()}  (论文 4510 / 我侧 4505)")
print(f"  唯一蛋白: {all_df['Protein'].nunique()}  (论文 2181)")
print(f"  总对(n): {total}  (27,457 去重后)")

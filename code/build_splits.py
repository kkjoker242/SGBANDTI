# -*- coding: utf-8 -*-
"""
数据去重与重新划分脚本（项6 + 项10）。

用途：
- 读取已有划分（如 datasets/biosnap/random/*.csv）合并为全量交互集合；
- 规范化 SMILES（RDKit canonical）与蛋白序列（大写）；
- 按 (SMILES, Protein) 去重，处理冲突标签；
- 重新生成 7:1:2 的 random 划分，以及 drug-side cold-start（unseen_drug）
  与 target-side cold-start（unseen_target）划分；
- 默认 dry-run 输出统计，加 --apply 才写入 datasets/。

示例：
  python build_splits.py --data biosnap --seed 42
  python build_splits.py --data biosnap --seed 42 --apply
"""
import argparse
import os
import random
import numpy as np
import pandas as pd
from rdkit import Chem

parser = argparse.ArgumentParser(description="Deduplicate and regenerate dataset splits")
parser.add_argument("--data", default="biosnap", choices=["biosnap", "bindingdb"])
parser.add_argument("--source-split", default="random", help="source split dir to merge (e.g. random)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--apply", action="store_true", help="write split files (default: dry-run report)")
parser.add_argument("--skip-random", action="store_true",
                    help="只重生成冷启动划分（unseen_drug/unseen_target），不动 random（避免覆盖已用结果）")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

data_dir = os.path.join("datasets", args.data)
src_dir = os.path.join(data_dir, args.source_split)


def load_full_set():
    """合并 source split 的 train/val/test 为全量交互集。"""
    frames = []
    for s in ["train", "val", "test"]:
        p = os.path.join(src_dir, f"{s}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"no split files under {src_dir}")
    return pd.concat(frames, ignore_index=True)


def canonical_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def prepare(df):
    d = df[["SMILES", "Protein", "Y"]].copy()
    d["Protein"] = d["Protein"].str.upper()
    d["smi_c"] = d["SMILES"].map(canonical_smiles)
    n_parse_fail = int(d["smi_c"].isna().sum())
    d = d.dropna(subset=["smi_c"])
    return d, n_parse_fail


def dedup(d):
    """按 (smi_c, Protein) 去重；冲突标签取正（Y=1 优先），并统计冲突数。"""
    # 冲突标签：同一 pair 出现不同 Y
    g = d.groupby(["smi_c", "Protein"])["Y"].agg(lambda s: list(s))
    conflict = int(g.map(lambda s: len(set(s)) > 1).sum())
    ded = d.sort_values("Y", ascending=False).drop_duplicates(
        subset=["smi_c", "Protein"], keep="first"
    ).reset_index(drop=True)
    return ded, conflict


def write_split(d, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    d = d[["SMILES", "Protein", "Y"]]
    d.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)


def rand_partition(n):
    """按 7:1:2 划分索引。"""
    idx = list(range(n))
    random.shuffle(idx)
    n_tr = int(round(n * 0.7))
    n_va = int(round(n * 0.1))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def split_by_entity(d, key):
    """按实体分组冷启动：实体互斥的 train/val/test。

    key 应为规范化后的实体列（canonical SMILES=smi_c 或 标准化蛋白=Protein 大写），
    避免同一化合物不同 SMILES 写法跨集合泄漏。输出仍保留原始 SMILES 列。
    """
    entities = d[key].unique().tolist()
    random.shuffle(entities)
    n_tr = int(round(len(entities) * 0.7))
    n_va = int(round(len(entities) * 0.1))
    tr_e, va_e, te_e = set(entities[:n_tr]), set(entities[n_tr:n_tr + n_va]), set(entities[n_tr + n_va:])
    tr = d[d[key].isin(tr_e)]
    va = d[d[key].isin(va_e)]
    te = d[d[key].isin(te_e)]
    # 校验：按规范化实体分组后，train/val/test 两两交集必须为 0（无实体泄漏）
    inter_tv = len(tr_e & va_e)
    inter_tt = len(tr_e & te_e)
    inter_vt = len(va_e & te_e)
    total_inter = inter_tv + inter_tt + inter_vt
    print(f"  实体交集校验（{key}）: train-val={inter_tv} train-test={inter_tt} val-test={inter_vt}，合计={total_inter}")
    assert total_inter == 0, f"实体泄漏！{key} 跨集合交集非 0"
    return tr, va, te


def report_stats(tag, splits):
    tr, va, te = splits
    all_df = pd.concat(splits, ignore_index=True)
    print(f"--- {tag} ---")
    for name, df in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {name:6s}: {len(df):6d} | drugs={df.SMILES.nunique():5d} proteins={df.Protein.nunique():5d} | Y1={int((df.Y==1).sum()):5d} Y0={int((df.Y==0).sum()):5d}")
    tr_d = set(tr.SMILES); te_d = set(te.SMILES)
    tr_p = set(tr.Protein); te_p = set(te.Protein)
    print(f"  test 药物已见于训练: {len(te_d & tr_d)}/{len(te_d)} | test 蛋白已见于训练: {len(te_p & tr_p)}/{len(te_p)}")
    return all_df


def main():
    raw = load_full_set()
    d, n_parse_fail = prepare(raw)
    ded, n_conflict = dedup(d)
    # 排序保证确定性：无论输入行序如何，重跑结果一致（避免从自身输出读入时划分漂移）
    ded = ded.sort_values(["smi_c", "Protein"]).reset_index(drop=True)

    print(f"=== {args.data}（源 {args.source_split}）===")
    print(f"原始样本: {len(raw)} | 规范化为 {len(d)}（SMILES 解析失败 {n_parse_fail}）| 去重后 {len(ded)}（冲突标签 {n_conflict}）")

    if args.apply:
        out_root = data_dir
    else:
        out_root = os.path.join(data_dir, "_dry_run")
        print("(dry-run，不写盘；加 --apply 生效)")

    # 1) random 7:1:2（--skip-random 时跳过，避免覆盖已跑结果所用的划分）
    if not args.skip_random:
        tr_i, va_i, te_i = rand_partition(len(ded))
        r_tr, r_va, r_te = ded.iloc[tr_i], ded.iloc[va_i], ded.iloc[te_i]
        report_stats("random", (r_tr, r_va, r_te))
        if args.apply:
            write_split(r_tr, "train", os.path.join(out_root, "random"))
            write_split(r_va, "val", os.path.join(out_root, "random"))
            write_split(r_te, "test", os.path.join(out_root, "random"))
    else:
        print("(--skip-random 已设，跳过 random 划分)")

    # 2) unseen_drug（药物互斥冷启动）——按 canonical SMILES 分组防泄漏
    u_tr, u_va, u_te = split_by_entity(ded, "smi_c")
    report_stats("unseen_drug", (u_tr, u_va, u_te))
    if args.apply:
        write_split(u_tr, "train", os.path.join(out_root, "unseen_drug"))
        write_split(u_va, "val", os.path.join(out_root, "unseen_drug"))
        write_split(u_te, "test", os.path.join(out_root, "unseen_drug"))

    # 3) unseen_target（蛋白互斥冷启动）
    t_tr, t_va, t_te = split_by_entity(ded, "Protein")
    report_stats("unseen_target", (t_tr, t_va, t_te))
    if args.apply:
        write_split(t_tr, "train", os.path.join(out_root, "unseen_target"))
        write_split(t_va, "val", os.path.join(out_root, "unseen_target"))
        write_split(t_te, "test", os.path.join(out_root, "unseen_target"))

    print(f"\n完成。{'已写入 ' + out_root if args.apply else 'dry-run 报告完毕'}")


if __name__ == "__main__":
    main()

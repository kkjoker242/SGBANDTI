# -*- coding: utf-8 -*-
"""
冻结最终 split（P0-#6）：记录每个 split CSV 的 md5 哈希、实体/正负统计与当前 git commit，
输出到 SPLITS_FROZEN.json 与 SPLITS_FROZEN.md，保证数据版本可审计、可复现。

用法：
  python freeze_splits.py
"""
import glob
import hashlib
import json
import os
import subprocess

import pandas as pd

SPLITS = [
    ("bindingdb", "random"),
    ("biosnap", "random"),
    ("biosnap", "unseen_drug"),
    ("biosnap", "unseen_target"),
]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "N/A"


def main():
    commit = git_head()
    records = {}
    for ds, sp in SPLITS:
        d = os.path.join("datasets", ds, sp)
        if not os.path.isdir(d):
            print(f"跳过（不存在）: {d}")
            continue
        rec = {"hash": {}, "counts": {}}
        frames = []
        for s in ["train", "val", "test"]:
            p = os.path.join(d, f"{s}.csv")
            if os.path.isfile(p):
                rec["hash"][s] = md5(p)
                df = pd.read_csv(p)
                frames.append((s, df))
        all_df = pd.concat([f for _, f in frames], ignore_index=True)
        rec["counts"] = {
            "samples": {s: len(f) for s, f in frames},
            "drugs": int(all_df["SMILES"].nunique()),
            "proteins": int(all_df["Protein"].nunique()),
            "positive": int((all_df["Y"] == 1).sum()),
            "negative": int((all_df["Y"] == 0).sum()),
        }
        records[f"{ds}/{sp}"] = rec

    out = {"git_commit": commit, "note": "冻结的最终 split；任何数据变化都会导致哈希改变", "splits": records}
    with open("SPLITS_FROZEN.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    with open("SPLITS_FROZEN.md", "w", encoding="utf-8") as f:
        f.write("# SPLITS_FROZEN（最终数据划分冻结记录）\n\n")
        f.write(f"Git commit: `{commit}`\n\n")
        f.write("| 划分 | split 样本数(t/v/te) | 药物 | 蛋白 | 正/负 | md5(train/val/test) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for k, r in records.items():
            c = r["counts"]
            f.write(f"| {k} | {c['samples']['train']}/{c['samples']['val']}/{c['samples']['test']} "
                    f"| {c['drugs']} | {c['proteins']} | {c['positive']}/{c['negative']} "
                    f"| {r['hash']['train'][:8]}/{r['hash']['val'][:8]}/{r['hash']['test'][:8]} |\n")

    print(f"git commit: {commit}")
    for k, r in records.items():
        c = r["counts"]
        print(f"{k}: {c['samples']['train']}/{c['samples']['val']}/{c['samples']['test']} 样本, "
              f"{c['drugs']} 药物, {c['proteins']} 蛋白, 正/负 {c['positive']}/{c['negative']}")
    print("已输出 SPLITS_FROZEN.json 和 SPLITS_FROZEN.md")


if __name__ == "__main__":
    main()

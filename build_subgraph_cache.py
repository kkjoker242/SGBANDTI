import argparse
import os

import pandas as pd
import torch

from dataloader import build_cache_samples


parser = argparse.ArgumentParser(description="Precompute NestedGNN subgraph caches")
parser.add_argument(
    "--data",
    default="biosnap",
    type=str,
    metavar="TASK",
    help="dataset",
    choices=["bindingdb", "biosnap", "human"],
)
parser.add_argument(
    "--split",
    default="random",
    type=str,
    metavar="S",
    help="split task",
    choices=["random", "cold", "cluster", "unseen_drug"],
)
parser.add_argument(
    "--hop",
    default=2,
    type=int,
    metavar="H",
    help="k-hop subgraph size for NestedGNN cache",
)
parser.add_argument(
    "--num-workers",
    default=1,
    type=int,
    metavar="N",
    help="number of parallel workers used to build cache",
)
parser.add_argument(
    "--worker-type",
    default="process",
    type=str,
    choices=["process", "thread"],
    help="parallel backend for cache generation",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="rebuild cache even if target file already exists",
)
args = parser.parse_args()


def get_cache_path(split_file_name):
    cache_dir = os.path.join(
        "./datasets",
        "subgraph_cache",
        args.data,
        args.split,
        f"hop_{args.hop}",
    )
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{split_file_name}.pt")


def build_one_split(data_dir, split_file_name):
    csv_path = os.path.join(data_dir, f"{split_file_name}.csv")
    if not os.path.exists(csv_path):
        print(f"Skip {split_file_name}: file not found at {csv_path}")
        return

    cache_path = get_cache_path(split_file_name)
    if os.path.exists(cache_path) and not args.force:
        print(f"Skip {split_file_name}: cache already exists at {cache_path}")
        return

    df = pd.read_csv(csv_path)
    print(
        f"Building {split_file_name} cache with {len(df)} samples "
        f"using {args.worker_type} x {args.num_workers}"
    )
    cached_samples = build_cache_samples(
        df,
        max_drug_nodes=290,
        use_nested=True,
        h=args.hop,
        num_workers=args.num_workers,
        worker_type=args.worker_type,
        show_progress=True,
        progress_desc=f"{split_file_name} hop={args.hop}",
    )
    torch.save(cached_samples, cache_path)
    print(f"Saved {split_file_name} cache to {cache_path}")


def main():
    data_dir = os.path.join("./datasets", args.data, args.split)
    print(f"Preparing subgraph cache for dataset={args.data}, split={args.split}, hop={args.hop}")
    print(f"Source directory: {data_dir}")
    print(f"Parallel setup: {args.worker_type} x {args.num_workers}")

    for split_file_name in ["train", "val", "test"]:
        build_one_split(data_dir, split_file_name)

    print("Cache generation finished.")


if __name__ == "__main__":
    main()

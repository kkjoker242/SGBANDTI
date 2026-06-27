import argparse
import os
import re
from time import time

import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from dataloader import DTIDataset, collate_fn_nested
from models import NGNN_BAN
from trainer import Trainer
from utils import mkdir, set_seed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Evaluate a saved NGNN_BAN checkpoint on the test set")
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
    "--seed",
    default=42,
    type=int,
    help="seed used during training; used to locate the saved checkpoint when --checkpoint is not provided",
)
parser.add_argument(
    "--checkpoint",
    default=None,
    type=str,
    help="path to a saved model checkpoint (.pth/.pt). If omitted, the script auto-loads the best model for the given data/split/hop/seed",
)
args = parser.parse_args()


def find_best_checkpoint():
    checkpoint_dir = os.path.join(
        "./result",
        f"{args.data}_{args.split}_hop{args.hop}",
        f"seed_{args.seed}",
    )
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    pattern = re.compile(r"best_model_epoch_(\d+)\.(pth|pt)$")
    candidates = []
    for file_name in os.listdir(checkpoint_dir):
        match = pattern.match(file_name)
        if match:
            candidates.append((int(match.group(1)), os.path.join(checkpoint_dir, file_name)))

    if not candidates:
        raise FileNotFoundError(
            f"No best-model checkpoint found in {checkpoint_dir}. "
            f"Expected files like best_model_epoch_XX.pth"
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_checkpoint():
    checkpoint_path = args.checkpoint or find_best_checkpoint()
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return checkpoint_path


def load_model_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        if "test_metrics" in checkpoint and "config" in checkpoint:
            raise ValueError(
                f"{checkpoint_path} looks like result_metrics.pt, which only stores metrics and config, not model weights. "
                f"Please pass best_model_epoch_*.pth or model_epoch_*.pth instead."
            )
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

    model.load_state_dict(state_dict)
    return model


def build_test_dataloader(cfg):
    data_folder = os.path.join(f"./datasets/{args.data}", str(args.split))
    test_path = os.path.join(data_folder, "test.csv")
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    df_test = pd.read_csv(test_path)
    test_dataset = DTIDataset(
        df_test.index.values,
        df_test,
        dataset_name=args.data,
        split_name=args.split,
        split_file_name="test",
        h=args.hop,
    )

    params = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": False,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": False,
        "collate_fn": collate_fn_nested,
    }
    return DataLoader(test_dataset, **params)


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    cfg = get_cfg_defaults()
    cfg.SOLVER.SEED = args.seed
    set_seed(args.seed)

    checkpoint_path = resolve_checkpoint()
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cfg.RESULT.OUTPUT_DIR = checkpoint_dir
    mkdir(cfg.RESULT.OUTPUT_DIR)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Random seed: {args.seed}")
    print(f"Running on: {device}", end="\n\n")

    test_dataloader = build_test_dataloader(cfg)

    model = NGNN_BAN(**cfg).to(device)
    model = load_model_weights(model, checkpoint_path)

    trainer = Trainer(
        model,
        optim=None,
        device=device,
        train_dataloader=test_dataloader,
        val_dataloader=test_dataloader,
        test_dataloader=test_dataloader,
        **cfg,
    )
    trainer.best_model = model
    trainer.best_epoch = None

    auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = trainer.test(
        dataloader="test"
    )

    result = {
        "checkpoint": checkpoint_path,
        "seed": args.seed,
        "auroc": auroc,
        "auprc": auprc,
        "test_loss": test_loss,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "thred_optim": thred_optim,
        "F1": f1,
        "Precision": precision,
    }

    result_path = os.path.join(checkpoint_dir, "checkpoint_test_metrics.pt")
    torch.save(result, result_path)

    print("Test metrics:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"\nSaved test metrics to: {result_path}")


if __name__ == "__main__":
    start = time()
    main()
    end = time()
    print(f"Total running time: {round(end - start, 2)}s")

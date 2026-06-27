from models import NGNN_BAN
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, collate_fn_nested
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings
import os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 52, 62, 72, 82]

parser = argparse.ArgumentParser(description="NGNN_BAN for DTI prediction")
parser.add_argument('--data', default='biosnap', type=str, metavar='TASK',
                    help='dataset', choices=['bindingdb', 'biosnap', 'human'])
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster', 'unseen_drug'])
parser.add_argument('--hop', default=2, type=int, metavar='H', help='k-hop subgraph size for NestedGNN cache')
args = parser.parse_args()


def run_single_seed(seed, df_train, df_val, df_test):
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.SOLVER.SEED = seed
    set_seed(seed)

    run_output_dir = os.path.join(
        cfg.RESULT.OUTPUT_DIR,
        f"{args.data}_{args.split}_hop{args.hop}",
        f"seed_{seed}",
    )
    cfg.RESULT.OUTPUT_DIR = run_output_dir
    mkdir(cfg.RESULT.OUTPUT_DIR)

    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Random seed: {seed}")
    print(f"Running on: {device}", end="\n\n")

    train_dataset = DTIDataset(
        df_train.index.values,
        df_train,
        dataset_name=args.data,
        split_name=args.split,
        split_file_name='train',
        h=args.hop,
    )
    val_dataset = DTIDataset(
        df_val.index.values,
        df_val,
        dataset_name=args.data,
        split_name=args.split,
        split_file_name='val',
        h=args.hop,
    )
    test_dataset = DTIDataset(
        df_test.index.values,
        df_test,
        dataset_name=args.data,
        split_name=args.split,
        split_file_name='test',
        h=args.hop,
    )

    params = {
        'batch_size': cfg.SOLVER.BATCH_SIZE,
        'shuffle': True,
        'num_workers': cfg.SOLVER.NUM_WORKERS,
        'drop_last': True,
        'collate_fn': collate_fn_nested
    }

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    model = NGNN_BAN(**cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    trainer = Trainer(
        model, opt, device,
        training_generator,
        val_generator,
        test_generator,
        **cfg
    )

    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


def main():
    data_folder = os.path.join(f"./datasets/{args.data}", str(args.split))
    train_path = os.path.join(data_folder, "train.csv")
    val_path = os.path.join(data_folder, "val.csv")
    test_path = os.path.join(data_folder, "test.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    all_results = []
    for seed in SEEDS:
        print(f"\n{'=' * 20} Seed {seed} {'=' * 20}")
        result = run_single_seed(seed, df_train, df_val, df_test)
        result["seed"] = seed
        all_results.append(result)

    summary_dir = os.path.join("./result", f"{args.data}_{args.split}_hop{args.hop}")
    mkdir(summary_dir)

    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df[
        [
            "seed",
            "auroc",
            "auprc",
            "test_loss",
            "sensitivity",
            "specificity",
            "accuracy",
            "thred_optim",
            "best_epoch",
            "F1",
            "Precision",
        ]
    ]
    summary_path = os.path.join(summary_dir, "seed_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    numeric_cols = [col for col in summary_df.columns if col != "seed"]
    stats_df = summary_df[numeric_cols].agg(["mean", "std"])
    stats_path = os.path.join(summary_dir, "seed_summary_stats.csv")
    stats_df.to_csv(stats_path)

    print("\nSeed summary:")
    print(summary_df.to_string(index=False))
    print("\nSeed statistics:")
    print(stats_df.to_string())
    print(f"\nSaved seed summary to: {summary_path}")
    print(f"Saved seed statistics to: {stats_path}")

    return all_results


if __name__ == '__main__':
    s = time()
    results = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")

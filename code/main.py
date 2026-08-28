from models import SGBANDTI
from time import time
from utils import set_seed, mkdir
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

parser = argparse.ArgumentParser(description="SGBANDTI for DTI prediction")
parser.add_argument('--data', default='biosnap', type=str, metavar='TASK',
                    help='dataset', choices=['bindingdb', 'biosnap', 'human'])
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster', 'unseen_drug', 'unseen_target'])
parser.add_argument('--hop', default=2, type=int, metavar='H', help='k-hop subgraph size for NestedGNN cache')
parser.add_argument('--ablation', default='full', type=str, metavar='A',
                    help='ablation: full | no_subgraph | no_ban | no_both',
                    choices=['full', 'no_subgraph', 'no_ban', 'no_both'])
parser.add_argument('--seeds', default=None, type=str, metavar='S',
                    help="comma-separated seed list to override defaults, e.g. 82 或 42,52")
parser.add_argument('--dropout', default=None, type=float, metavar='P',
                    help="override DECODER.DROPOUT (MLP decoder dropout, default 0.0)")
parser.add_argument('--heads', default=None, type=int, metavar='H',
                    help="override BCN.HEADS (BAN attention heads, default 2)")
parser.add_argument('--weight-decay', default=None, type=float, metavar='WD',
                    help="override SOLVER.WEIGHT_DECAY (Adam weight decay, default 0.0)")
parser.add_argument('--lr', default=None, type=float, metavar='LR',
                    help="override SOLVER.LR (Adam learning rate, default 5e-5)")
parser.add_argument('--hidden', default=None, type=int, metavar='H',
                    help="override DRUG.HIDDEN_LAYERS + PROTEIN.NUM_FILTERS width (default 128)")
parser.add_argument('--max-epoch', default=None, type=int, metavar='E',
                    help="override SOLVER.MAX_EPOCH (default 150)")
parser.add_argument('--cosine', action='store_true',
                    help="use CosineAnnealingLR scheduler (T_max=MAX_EPOCH)")
parser.add_argument('--batch-size', default=None, type=int, metavar='B',
                    help="override SOLVER.BATCH_SIZE (default 64)")
parser.add_argument('--tag', default=None, type=str, metavar='T',
                    help="output dir suffix to isolate tuning runs, e.g. tune_dropout02_wd1e4")
args = parser.parse_args()

if args.seeds:
    SEEDS = [int(s) for s in args.seeds.split(',')]
    print(f"Overridden seeds: {SEEDS}")


def apply_ablation(cfg):
    if args.ablation in ('no_subgraph', 'no_both'):
        cfg.ABLATION.USE_SUBGRAPH = False
    if args.ablation in ('no_ban', 'no_both'):
        cfg.ABLATION.USE_BAN = False
    return cfg


def run_single_seed(seed, df_train, df_val, df_test):
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg = apply_ablation(cfg)
    cfg.SOLVER.SEED = seed
    if args.dropout is not None:
        cfg.DECODER.DROPOUT = args.dropout
    if args.weight_decay is not None:
        cfg.SOLVER.WEIGHT_DECAY = args.weight_decay
    if args.heads is not None:
        cfg.BCN.HEADS = args.heads
    if args.lr is not None:
        cfg.SOLVER.LR = args.lr
    if args.hidden is not None:
        cfg.DRUG.HIDDEN_LAYERS = args.hidden
        cfg.PROTEIN.NUM_FILTERS = [args.hidden, args.hidden, args.hidden]
    if args.max_epoch is not None:
        cfg.SOLVER.MAX_EPOCH = args.max_epoch
    if args.cosine:
        cfg.SOLVER.COSINE = True
    if args.batch_size is not None:
        cfg.SOLVER.BATCH_SIZE = args.batch_size
    set_seed(seed)

    ablation_tag = "" if args.ablation == "full" else f"_{args.ablation}"
    tune_tag = "" if args.tag is None else f"_{args.tag}"
    run_output_dir = os.path.join(
        cfg.RESULT.OUTPUT_DIR,
        f"{args.data}_{args.split}_hop{args.hop}{ablation_tag}{tune_tag}",
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
        use_nested=cfg.ABLATION.USE_SUBGRAPH,
    )
    val_dataset = DTIDataset(
        df_val.index.values,
        df_val,
        dataset_name=args.data,
        split_name=args.split,
        split_file_name='val',
        h=args.hop,
        use_nested=cfg.ABLATION.USE_SUBGRAPH,
    )
    test_dataset = DTIDataset(
        df_test.index.values,
        df_test,
        dataset_name=args.data,
        split_name=args.split,
        split_file_name='test',
        h=args.hop,
        use_nested=cfg.ABLATION.USE_SUBGRAPH,
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

    model = SGBANDTI(**cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    # 与 utils.set_seed 的 cudnn.deterministic=True / benchmark=False 保持一致，
    # 保证训练可复现（避免非确定性快速算法）
    torch.backends.cudnn.benchmark = False

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

    ablation_tag = "" if args.ablation == "full" else f"_{args.ablation}"
    tune_tag = "" if args.tag is None else f"_{args.tag}"
    summary_dir = os.path.join("./result", f"{args.data}_{args.split}_hop{args.hop}{ablation_tag}{tune_tag}")
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

comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
from models import DrugBAN
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster', 'unseen_drug', 'unseen_target'])
parser.add_argument('--seeds', default=None, type=str, metavar='SEEDS',
                    help="comma-separated seeds, e.g. 82 or 42,52,62,72,82")
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else [42, 52, 62, 72, 82]
    base_out = cfg.RESULT.OUTPUT_DIR
    print(f"Config yaml: {args.cfg}")
    print(f"Seeds: {seeds}")
    print(f"Running on: {device}", end="\n\n")

    all_results = []
    for seed in seeds:
        print(f"\n{'='*50}\nSeed: {seed}\n{'='*50}")
        torch.cuda.empty_cache()
        set_seed(seed)
        cfg.SOLVER.SEED = seed
        cfg.RESULT.OUTPUT_DIR = os.path.join(base_out, f"seed_{seed}")
        mkdir(cfg.RESULT.OUTPUT_DIR)
        experiment = None

        dataFolder = os.path.join('./datasets', args.data, str(args.split))

        if not cfg.DA.TASK:
            train_path = os.path.join(dataFolder, 'train.csv')
            val_path = os.path.join(dataFolder, "val.csv")
            test_path = os.path.join(dataFolder, "test.csv")
            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)
            df_test = pd.read_csv(test_path)
            train_dataset = DTIDataset(df_train.index.values, df_train)
            val_dataset = DTIDataset(df_val.index.values, df_val)
            test_dataset = DTIDataset(df_test.index.values, df_test)
        else:
            train_source_path = os.path.join(dataFolder, 'source_train.csv')
            train_target_path = os.path.join(dataFolder, 'target_train.csv')
            test_target_path = os.path.join(dataFolder, 'target_test.csv')
            df_train_source = pd.read_csv(train_source_path)
            df_train_target = pd.read_csv(train_target_path)
            df_test_target = pd.read_csv(test_target_path)
            train_dataset = DTIDataset(df_train_source.index.values, df_train_source)
            train_target_dataset = DTIDataset(df_train_target.index.values, df_train_target)
            test_target_dataset = DTIDataset(df_test_target.index.values, df_test_target)

        params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                  'drop_last': True, 'collate_fn': graph_collate_func}

        if not cfg.DA.USE:
            training_generator = DataLoader(train_dataset, **params)
            params['shuffle'] = False
            params['drop_last'] = False
            if not cfg.DA.TASK:
                val_generator = DataLoader(val_dataset, **params)
                test_generator = DataLoader(test_dataset, **params)
            else:
                val_generator = DataLoader(test_target_dataset, **params)
                test_generator = DataLoader(test_target_dataset, **params)
        else:
            source_generator = DataLoader(train_dataset, **params)
            target_generator = DataLoader(train_target_dataset, **params)
            n_batches = max(len(source_generator), len(target_generator))
            multi_generator = MultiDataLoader(dataloaders=[source_generator, target_generator], n_batches=n_batches)
            params['shuffle'] = False
            params['drop_last'] = False
            val_generator = DataLoader(test_target_dataset, **params)
            test_generator = DataLoader(test_target_dataset, **params)

        model = DrugBAN(**cfg).to(device)

        if cfg.DA.USE:
            if cfg["DA"]["RANDOM_LAYER"]:
                domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
            else:
                domain_dmm = Discriminator(input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"],
                                           n_class=cfg["DECODER"]["BINARY"]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
            opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
            opt_da = None
            domain_dmm = None

        torch.backends.cudnn.benchmark = True

        if not cfg.DA.USE:
            trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, opt_da=None,
                              discriminator=None, experiment=experiment, **cfg)
        else:
            trainer = Trainer(model, opt, device, multi_generator, val_generator, test_generator, opt_da=opt_da,
                              discriminator=domain_dmm, experiment=experiment, **cfg)
        result = trainer.train()
        result["seed"] = seed
        all_results.append(result)

        with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
            wf.write(str(model))
        print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    summary = pd.DataFrame(all_results)
    print("\n=== Seed Summary ===")
    print(summary.to_string(index=False))
    return all_results


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")

"""Re-evaluate saved DrugBAN best models with STANDARD metric formulas.

Usage:
    python evaluate_saved_model.py --result_dir result_biosnap --data biosnap --split random --seeds 42,52,62,72,82
"""
import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, precision_recall_curve,
                             confusion_matrix)
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from dataloader import DTIDataset
from models import DrugBAN, binary_cross_entropy
from utils import graph_collate_func

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, type=str)
parser.add_argument('--data', required=True, type=str)
parser.add_argument('--split', default='random', type=str)
parser.add_argument('--seeds', default='42,52,62,72,82', type=str)
parser.add_argument('--cfg', default='configs/DrugBAN.yaml', type=str)
parser.add_argument('--batch_size', default=64, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = get_cfg_defaults()
cfg.merge_from_file(args.cfg)

data_folder = os.path.join('./datasets', args.data, args.split)
df_test = pd.read_csv(os.path.join(data_folder, 'test.csv'))
test_dataset = DTIDataset(df_test.index.values, df_test)
test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, drop_last=False, collate_fn=graph_collate_func)

seeds = [int(s) for s in args.seeds.split(',')]
rows = []
for seed in seeds:
    seed_dir = os.path.join(args.result_dir, f'seed_{seed}')
    ckpt_files = glob.glob(os.path.join(seed_dir, 'best_model_epoch_*.pth'))
    if not ckpt_files:
        print(f'[seed {seed}] no best model found in {seed_dir}, skipped')
        continue
    ckpt_path = sorted(ckpt_files)[-1]
    model = DrugBAN(**cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()

    y_label, y_pred = [], []
    with torch.no_grad():
        for v_d, v_p, labels in test_generator:
            v_d, v_p = v_d.to(device), v_p.to(device)
            labels = labels.float().to(device)
            _, _, _, score = model(v_d, v_p)
            n, _ = binary_cross_entropy(score, labels)
            y_label += labels.cpu().tolist()
            y_pred += n.cpu().tolist()

    auroc = roc_auc_score(y_label, y_pred)
    auprc = average_precision_score(y_label, y_pred)
    prec, recall, pr_thr = precision_recall_curve(y_label, y_pred)
    f1_pr = 2 * prec * recall / (prec + recall + 1e-12)
    thred_optim = pr_thr[np.argmax(f1_pr[:-1])]
    y_pred_s = [1 if i >= thred_optim else 0 for i in y_pred]
    cm = confusion_matrix(y_label, y_pred_s)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    f1 = f1_score(y_label, y_pred_s)
    sens = recall_score(y_label, y_pred_s)
    spec = tn / (tn + fp)
    acc = (tn + tp) / (tn + fp + fn + tp)
    prec_score = precision_score(y_label, y_pred_s)

    print(f'[seed {seed}] model: {os.path.basename(ckpt_path)}  threshold={thred_optim:.4f}')
    print(f'  AUROC {auroc:.4f}  AUPRC {auprc:.4f}  F1 {f1:.4f}  Sens {sens:.4f}  Spe {spec:.4f}  Acc {acc:.4f}  Precision {prec_score:.4f}')
    rows.append(dict(seed=seed, auroc=auroc, auprc=auprc, f1=f1, sens=sens, spe=spec, acc=acc, prec=prec_score, thr=thred_optim))
    np.save(os.path.join(seed_dir, 'test_y_pred.npy'), np.asarray(y_pred))
    np.save(os.path.join(seed_dir, 'test_y_label.npy'), np.asarray(y_label))
    del model
    torch.cuda.empty_cache()

print('\n=== Standard-metric re-evaluation summary ===')
df = pd.DataFrame(rows)
for col in ['auroc', 'auprc', 'f1', 'sens', 'spe', 'acc', 'prec']:
    print(f'{col}: {df[col].mean():.4f} +/- {df[col].std():.4f}')
out_csv = os.path.join(args.result_dir, 'standard_metrics_reeval.csv')
df.to_csv(out_csv, index=False)
print(f'\nsaved to {out_csv}')

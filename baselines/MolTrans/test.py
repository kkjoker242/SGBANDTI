import copy
import os
from time import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc, precision_recall_curve
from torch import nn
from torch.autograd import Variable
from torch.utils import data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(2)
np.random.seed(3)
from argparse import ArgumentParser
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = ArgumentParser(description='MolTrans Testing.')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--task', choices=['biosnap', 'bindingdb', 'GPCR', 'human', 'C.elegans'],
                    default='bindingdb', type=str, metavar='TASK',
                    help='Task name.')
parser.add_argument('--model-path', default='./model_checkpoints', type=str,
                    metavar='PATH',
                    help='Path to save the trained model')
parser.add_argument('--output-dir', default='./results', type=str,
                    metavar='DIR',
                    help='Directory to save results and plots')
parser.add_argument('--seeds', nargs='+', default=[42, 52, 62, 72, 82], type=int,
                    metavar='SEED', help='list of random seeds for multiple runs')


def get_task(task_name):
    if task_name.lower() == 'biosnap':
        return './dataset/BIOSNAP/random'
    elif task_name.lower() == 'bindingdb':
        return './dataset/bindingdb/random'
    elif task_name.lower() == 'GPCR':
        return './dataset/GPCR/random'
    elif task_name.lower() == 'human':
        return './dataset/human/random'
    elif task_name.lower() == 'C.elegans':
        return './dataset/C.elegans'


def test_with_metrics(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
        score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        logits_detach = logits.detach().cpu().numpy()
        label_ids = label.to('cpu').numpy()
        
        if logits_detach.shape[0] == label_ids.shape[0]:
            if logits.size() == label.size():
                loss = loss_fct(logits, label)
                loss_accumulate += loss
                count += 1
            
            y_pred = y_pred + logits_detach.flatten().tolist()
            y_label = y_label + label_ids.flatten().tolist()

    if count > 0:
        loss = loss_accumulate / count
    else:
        loss = torch.tensor(0.0)

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    precision, recall, pr_thresholds = precision_recall_curve(y_label, y_pred)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    
    valid_len = min(len(thresholds[5:]), len(f1_scores[5:]))
    if valid_len > 0:
        thred_optim = thresholds[5:5+valid_len][np.argmax(f1_scores[5:5+valid_len])]
    else:
        thred_optim = 0.5

    y_pred_s = [1 if i else 0 for i in (np.asarray(y_pred) >= thred_optim)]

    auc_k = auc(fpr, tpr)
    ap_score = average_precision_score(y_label, y_pred)

    cm1 = confusion_matrix(y_label, y_pred_s)

    total1 = sum(sum(cm1))
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    metrics = {
        'auroc': auc_k,
        'auprc': ap_score,
        'f1': f1_score(y_label, outputs),
        'loss': float(loss.item()) if isinstance(loss, torch.Tensor) else float(loss),
        'accuracy': accuracy1,
        'sensitivity': sensitivity1,
        'specificity': specificity1,
        'recall': recall_score(y_label, y_pred_s),
        'precision': precision_score(y_label, y_pred_s)
    }

    return y_pred, y_label, fpr, tpr, precision, recall, pr_thresholds, metrics


def plot_auroc(fpr, tpr, auroc, output_path):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (AUROC)', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUROC curve saved to: {output_path}")


def plot_auprc(precision, recall, auprc, output_path):
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUPRC = {auprc:.4f})')
    baseline = len([y for y in [1] * len(precision)]) / len(precision)
    plt.axhline(y=baseline, color='red', linestyle='--', lw=2, label='Random baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve (AUPRC)', fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUPRC curve saved to: {output_path}")


def plot_combined_curves_seeds(all_results, output_path):
    plt.figure(figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    plt.subplot(1, 2, 1)
    for i, (seed, fpr, tpr, auroc) in enumerate(all_results):
        plt.plot(fpr, tpr, color=colors[i], lw=1.5, alpha=0.7,
                 label=f'Seed {seed} (AUC = {auroc:.4f})')
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    valid_count = 0
    for seed, fpr, tpr, _ in all_results:
        if len(fpr) > 1 and len(tpr) > 1:
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            valid_count += 1
    if valid_count > 0:
        mean_tpr /= valid_count
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k-', lw=3, label=f'Mean (AUC = {mean_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('AUROC Curves (Multiple Seeds)', fontsize=14)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for i, (seed, precision, recall, auprc) in enumerate(all_results):
        if len(recall) > 0:
            plt.plot(recall, precision, color=colors[i], lw=1.5, alpha=0.7,
                     label=f'Seed {seed} (AUPRC = {auprc:.4f})')
    mean_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(mean_recall)
    valid_count = 0
    for seed, precision, recall, _ in all_results:
        if len(recall) > 1 and len(precision) > 1:
            mean_precision += np.interp(mean_recall, recall[::-1], precision[::-1])
            valid_count += 1
    if valid_count > 0:
        mean_precision /= valid_count
        mean_auprc = auc(mean_recall, mean_precision)
        plt.plot(mean_recall, mean_precision, 'k-', lw=3, label=f'Mean (AUPRC = {mean_auprc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('AUPRC Curves (Multiple Seeds)', fontsize=14)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined multi-seed curves saved to: {output_path}")


def save_metrics(metrics, output_path):
    with open(output_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Metrics saved to: {output_path}")


def main():
    config = BIN_config_DBPE()
    args = parser.parse_args()
    config['batch_size'] = args.batch_size

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    model = BIN_Interaction_Flat(**config)
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': args.workers,
              'drop_last': False}

    dataFolder = get_task(args.task)

    df_test = pd.read_csv(dataFolder + '/test.csv')
    df_test.columns = [col.strip() for col in df_test.columns]

    testing_set = BIN_Data_Encoder(df_test.index.values, df_test['Label'].values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    all_seeds_results = []
    all_metrics = []

    for seed in args.seeds:
        print(f'\n{"="*50}')
        print(f'>>> Evaluating model with seed = {seed} <<<')
        print(f'{"="*50}')

        model_file = os.path.join(args.model_path, f'best_model_{args.task}_seed{seed}.pt')
        if not os.path.exists(model_file):
            print(f"Model file not found: {model_file}, skipping...")
            continue

        print(f"Loading model from: {model_file}")
        model.load_state_dict(torch.load(model_file))

        print('--- Generating Predictions ---')
        with torch.set_grad_enabled(False):
            y_pred, y_label, fpr, tpr, precision, recall, pr_thresholds, metrics = test_with_metrics(testing_generator, model)

        print(f"\n--- Test Results (seed={seed}) ---")
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")

        all_seeds_results.append((seed, fpr, tpr, metrics['auroc'], precision, recall, metrics['auprc']))
        all_metrics.append(metrics)

        np.save(os.path.join(args.output_dir, f'y_pred_{args.task}_seed{seed}.npy'), np.array(y_pred))
        np.save(os.path.join(args.output_dir, f'y_label_{args.task}_seed{seed}.npy'), np.array(y_label))
        np.save(os.path.join(args.output_dir, f'fpr_{args.task}_seed{seed}.npy'), np.array(fpr))
        np.save(os.path.join(args.output_dir, f'tpr_{args.task}_seed{seed}.npy'), np.array(tpr))
        np.save(os.path.join(args.output_dir, f'precision_{args.task}_seed{seed}.npy'), np.array(precision))
        np.save(os.path.join(args.output_dir, f'recall_{args.task}_seed{seed}.npy'), np.array(recall))
        np.save(os.path.join(args.output_dir, f'pr_thresholds_{args.task}_seed{seed}.npy'), np.array(pr_thresholds))

        save_metrics(metrics, os.path.join(args.output_dir, f'metrics_{args.task}_seed{seed}.txt'))

        auroc_path = os.path.join(args.output_dir, f'auroc_curve_{args.task}_seed{seed}.png')
        plot_auroc(fpr, tpr, metrics['auroc'], auroc_path)

        auprc_path = os.path.join(args.output_dir, f'auprc_curve_{args.task}_seed{seed}.png')
        plot_auprc(precision, recall, metrics['auprc'], auprc_path)

    if not all_seeds_results:
        print('No model files found!')
        return

    print(f'\n{"="*60}')
    print(f'>>> Summary across {len(all_seeds_results)} seeds <<<')
    print(f'{"="*60}')

    aurocs = [m['auroc'] for m in all_metrics]
    auprcs = [m['auprc'] for m in all_metrics]
    f1s = [m['f1'] for m in all_metrics]

    print(f"AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    print(f"F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    summary_metrics = {
        'auroc_mean': float(np.mean(aurocs)),
        'auroc_std': float(np.std(aurocs)),
        'auprc_mean': float(np.mean(auprcs)),
        'auprc_std': float(np.std(auprcs)),
        'f1_mean': float(np.mean(f1s)),
        'f1_std': float(np.std(f1s)),
        'num_seeds': len(all_metrics)
    }
    save_metrics(summary_metrics, os.path.join(args.output_dir, f'metrics_summary_{args.task}.txt'))

    combined_path = os.path.join(args.output_dir, f'combined_curves_seeds_{args.task}.png')
    auroc_results = [(r[0], r[1], r[2], r[3]) for r in all_seeds_results]
    auprc_results = [(r[0], r[4], r[5], r[6]) for r in all_seeds_results]
    plot_combined_curves_seeds(auroc_results, combined_path)
    plot_combined_curves_seeds(auprc_results, os.path.join(args.output_dir, f'pr_curves_seeds_{args.task}.png'))

    np.save(os.path.join(args.output_dir, f'fpr_all_seeds_{args.task}.npy'),
            np.array([r[1] for r in all_seeds_results], dtype=object))
    np.save(os.path.join(args.output_dir, f'tpr_all_seeds_{args.task}.npy'),
            np.array([r[2] for r in all_seeds_results], dtype=object))
    np.save(os.path.join(args.output_dir, f'precision_all_seeds_{args.task}.npy'),
            np.array([r[4] for r in all_seeds_results], dtype=object))
    np.save(os.path.join(args.output_dir, f'recall_all_seeds_{args.task}.npy'),
            np.array([r[5] for r in all_seeds_results], dtype=object))
    np.save(os.path.join(args.output_dir, f'aurocs_{args.task}.npy'), np.array(aurocs))
    np.save(os.path.join(args.output_dir, f'auprcs_{args.task}.npy'), np.array(auprcs))

    print(f"\nAll data files saved to: {args.output_dir}")
    print(f"Files generated (per-seed):")
    for seed in args.seeds:
        print(f"  - y_pred_{args.task}_seed{seed}.npy")
        print(f"  - metrics_{args.task}_seed{seed}.txt")
        print(f"  - auroc_curve_{args.task}_seed{seed}.png")
        print(f"  - auprc_curve_{args.task}_seed{seed}.png")
    print(f"\nSummary files:")
    print(f"  - metrics_summary_{args.task}.txt")
    print(f"  - combined_curves_seeds_{args.task}.png")
    print(f"  - pr_curves_seeds_{args.task}.png")
    print(f"  - fpr_all_seeds_{args.task}.npy")
    print(f"  - tpr_all_seeds_{args.task}.npy")
    print(f"  - precision_all_seeds_{args.task}.npy")
    print(f"  - recall_all_seeds_{args.task}.npy")
    print(f"  - aurocs_{args.task}.npy")
    print(f"  - auprcs_{args.task}.npy")

    return all_metrics, summary_metrics


if __name__ == '__main__':
    main()

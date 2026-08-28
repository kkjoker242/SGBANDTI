from models import MGNDTI
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
import torch
import argparse
import warnings, os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, cfg, device):
    model = MGNDTI(**cfg).to(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, test_generator, device):
    model.eval()
    y_label, y_pred = [], []
    with torch.no_grad():
        for i, (v_s, v_d, v_p, labels) in enumerate(test_generator):
            v_s, v_d, v_p, labels = v_s.to(device), v_d.to(device), v_p.to(device), labels.float().to(device)
            v_d, v_s, v_p, f, score = model(v_s, v_d, v_p)
            y_label = y_label + labels.to("cpu").tolist()
            y_pred = y_pred + score.to("cpu").tolist()
    return np.array(y_label), np.array(y_pred)

def save_roc_pr_data(y_true, y_score, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    
    np.save(os.path.join(output_dir, "roc_fpr.npy"), fpr)
    np.save(os.path.join(output_dir, "roc_tpr.npy"), tpr)
    np.save(os.path.join(output_dir, "pr_precision.npy"), precision)
    np.save(os.path.join(output_dir, "pr_recall.npy"), recall)
    np.save(os.path.join(output_dir, "y_true.npy"), y_true)
    np.save(os.path.join(output_dir, "y_score.npy"), y_score)
    
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    
    return auroc, auprc

def plot_roc(fpr, tpr, auroc, output_dir, save=True):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'MGNDTI (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr(precision, recall, auprc, output_dir, save=True):
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'r-', linewidth=2, label=f'MGNDTI (AUPRC = {auprc:.4f})')
    baseline = np.sum(np.array(precision) > 0) / len(precision)
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="MGNDTI Test for ROC and AUPRC")
    parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='bindingdb')
    parser.add_argument('--model_path', type=str, help='path to model checkpoint', default=None)
    parser.add_argument('--epoch', type=int, help='model epoch to load', default=45)
    args = parser.parse_args()
    
    torch.cuda.empty_cache()
    cfg = get_cfg_defaults()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    
    output_path = os.path.join(cfg.RESULT.OUTPUT_DIR, args.data)
    mkdir(output_path)
    
    print(f"Running on: {device}", end="\n\n")
    dataFolder = f'../datasets/{args.data}/random/'
    
    test_set = pd.read_csv(dataFolder + "test.csv")
    print(f"test_set: {len(test_set)}")
    
    set_seed(cfg.SOLVER.SEED)
    test_dataset = DTIDataset(test_set.index.values, test_set)
    
    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': False, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': False, 'collate_fn': graph_collate_func}
    test_generator = DataLoader(test_dataset, **params)
    
    if args.model_path is None:
        model_path = os.path.join(output_path, f"best_model_epoch_{args.epoch}.pth")
    else:
        model_path = args.model_path
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, cfg, device)
    
    print("Running predictions...")
    y_true, y_score = predict(model, test_generator, device)
    
    print("Saving ROC and PR curve data...")
    auroc, auprc = save_roc_pr_data(y_true, y_score, output_path)
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    
    print("Generating plots...")
    fpr = np.load(os.path.join(output_path, "roc_fpr.npy"))
    tpr = np.load(os.path.join(output_path, "roc_tpr.npy"))
    precision = np.load(os.path.join(output_path, "pr_precision.npy"))
    recall = np.load(os.path.join(output_path, "pr_recall.npy"))
    
    plot_roc(fpr, tpr, auroc, output_path)
    plot_pr(precision, recall, auprc, output_path)
    
    print(f"\nNPY files saved to: {output_path}")
    print("  - roc_fpr.npy (False Positive Rate)")
    print("  - roc_tpr.npy (True Positive Rate)")
    print("  - pr_precision.npy (Precision)")
    print("  - pr_recall.npy (Recall)")
    print("  - y_true.npy (True labels)")
    print("  - y_score.npy (Prediction scores)")
    print("\nPNG files saved to:")
    print("  - roc_curve.png")
    print("  - pr_curve.png")

if __name__ == '__main__':
    main()

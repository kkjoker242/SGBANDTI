import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             recall_score, precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve, auc)

import utils

MAX_DRUG_LEN = 50       # ESPF 药物子词序列最大长度（与 drug2emb_encoder 一致）
MAX_PROTEIN_LEN = 545   # ESPF 蛋白子词序列最大长度（与 protein2emb_encoder 一致）

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# ESPF 编码（BPE 子词 -> token index 序列）
# ---------------------------------------------------------------------------

def _load_espf():
    utils._load_espf()

def encode_drug_espf(smiles):
    _load_espf()
    t1 = utils.dbpe.process_line(str(smiles)).split()
    try:
        i1 = np.asarray([utils.words2idx_d[i] for i in t1])
    except Exception:
        i1 = np.array([0])
    return i1

def encode_protein_espf(seq):
    _load_espf()
    t1 = utils.pbpe.process_line(str(seq)).split()
    try:
        i1 = np.asarray([utils.words2idx_p[i] for i in t1])
    except Exception:
        i1 = np.array([0])
    return i1

def pad_seq(idx_seq, max_len, pad_val):
    l = len(idx_seq)
    if l >= max_len:
        return idx_seq[:max_len]
    return np.pad(idx_seq, (0, max_len - l), 'constant', constant_values=pad_val)

# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------

class ESPF_Dataset(Dataset):
    """读取 DTI DataFrame（列：SMILES, Protein, Y），转为 ESPF 子词 token 序列。"""

    def __init__(self, df, drug_pad, protein_pad):
        self.drugs = []
        self.proteins = []
        self.labels = []
        for _, row in df.iterrows():
            self.drugs.append(pad_seq(encode_drug_espf(row['SMILES']), MAX_DRUG_LEN, drug_pad))
            self.proteins.append(pad_seq(encode_protein_espf(row['Protein']), MAX_PROTEIN_LEN, protein_pad))
            self.labels.append(float(row['Y']))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.LongTensor(self.drugs[idx]),
                torch.LongTensor(self.proteins[idx]),
                torch.FloatTensor([self.labels[idx]]))

# ---------------------------------------------------------------------------
# CNN1D 特征提取器（药物/蛋白质双分支共用）
# ---------------------------------------------------------------------------

class CNN1D_Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, filters, kernels, hidden_dim, max_len):
        super(CNN1D_Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=vocab_size)
        in_ch = [embed_dim] + filters
        layer_size = len(filters)
        self.conv = nn.ModuleList(
            [nn.Conv1d(in_ch[i], in_ch[i + 1], kernels[i]) for i in range(layer_size)])
        self.fc1 = nn.Linear(self._get_conv_output((embed_dim, max_len)), hidden_dim)

    def _get_conv_output(self, shape):
        bs = 1
        input_ = torch.rand(bs, *shape)
        output_feat = self._forward_features(input_)
        return output_feat.view(bs, -1).size(1)

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        x = self.embedding(v).permute(0, 2, 1)  # (B, embed_dim, seq_len)
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)

# ---------------------------------------------------------------------------
# ESPF + CNN1D 双分支 DTI 模型
# ---------------------------------------------------------------------------

class ESPF_CNN1D_DTI(nn.Module):
    def __init__(self, vocab_drug, vocab_protein, embed_dim=64,
                 cnn_filters=[32, 64, 128], cnn_kernels=[4, 4, 4],
                 hidden_dim=256, cls_hidden_dims=[1024, 1024, 512]):
        super(ESPF_CNN1D_DTI, self).__init__()
        self.drug_encoder = CNN1D_Encoder(vocab_drug, embed_dim, cnn_filters, cnn_kernels,
                                          hidden_dim, MAX_DRUG_LEN)
        self.protein_encoder = CNN1D_Encoder(vocab_protein, embed_dim, cnn_filters, cnn_kernels,
                                             hidden_dim, MAX_PROTEIN_LEN)

        dims = [hidden_dim * 2] + cls_hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, drug, protein):
        v_d = self.drug_encoder(drug)
        v_p = self.protein_encoder(protein)
        return self.classifier(torch.cat((v_d, v_p), 1))

# ---------------------------------------------------------------------------
# 训练 / 测试
# ---------------------------------------------------------------------------

def evaluate(model, loader):
    model.eval()
    y_pred_all, y_label_all = [], []
    with torch.no_grad():
        for drug, protein, label in loader:
            drug, protein, label = drug.to(device), protein.to(device), label.to(device)
            score = model(drug, protein)
            m = torch.nn.Sigmoid()
            logits = torch.squeeze(m(score)).detach().cpu().numpy()
            y_pred_all += logits.flatten().tolist()
            y_label_all += label.cpu().numpy().flatten().tolist()
    return np.asarray(y_pred_all), np.asarray(y_label_all)


def compute_metrics(y_pred, y_label):
    """返回 [AUROC, AUPRC, F1, Sensitivity, Specificity, Accuracy, Precision]"""
    y_pred_s = [1 if i else 0 for i in (y_pred >= 0.5)]
    tn, fp, fn, tp = confusion_matrix(y_label, y_pred_s).ravel()
    auroc = roc_auc_score(y_label, y_pred)
    auprc = average_precision_score(y_label, y_pred)
    f1 = f1_score(y_label, y_pred_s)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return auroc, auprc, f1, sensitivity, specificity, accuracy, precision


def train_espf_cnn(train_df, val_df, test_df, seed=42,
                   batch_size=64, lr=1e-4, train_epoch=50,
                   embed_dim=64, cnn_filters=[32, 64, 128], cnn_kernels=[4, 4, 4],
                   hidden_dim=256, cls_hidden_dims=[1024, 1024, 512],
                   patience=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    _load_espf()
    drug_pad = len(utils.words2idx_d)
    protein_pad = len(utils.words2idx_p)

    train_set = ESPF_Dataset(train_df, drug_pad, protein_pad)
    val_set = ESPF_Dataset(val_df, drug_pad, protein_pad)
    test_set = ESPF_Dataset(test_df, drug_pad, protein_pad)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ESPF_CNN1D_DTI(drug_pad, protein_pad, embed_dim=embed_dim,
                           cnn_filters=cnn_filters, cnn_kernels=cnn_kernels,
                           hidden_dim=hidden_dim, cls_hidden_dims=cls_hidden_dims)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_epoch, eta_min=0)
    loss_fct = torch.nn.BCELoss()
    m = torch.nn.Sigmoid()

    best_auc = 0.0
    best_model = copy.deepcopy(model)
    no_improve = 0

    for epo in range(train_epoch):
        model.train()
        total_loss = 0.0
        for drug, protein, label in train_loader:
            drug, protein, label = drug.to(device), protein.to(device), label.to(device)
            score = model(drug, protein)
            n = torch.squeeze(m(score), 1).squeeze()
            label = label.squeeze()
            loss = loss_fct(n, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        scheduler.step()

        # validation
        val_pred, val_label = evaluate(model, valid_loader)
        val_auc, val_auprc, val_f1, val_sen, val_spe, val_acc, val_pre = compute_metrics(val_pred, val_label)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = copy.deepcopy(model)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epo + 1}')
                break

        if (epo + 1) % 10 == 0 or epo == 0:
            print(f'Epoch {epo + 1}/{train_epoch}, Loss: {total_loss / len(train_loader):.4f}, '
                  f'Val AUROC: {val_auc:.4f}, Val AUPRC: {val_auprc:.4f}')

    # test with best model
    test_pred, test_label = evaluate(best_model, test_loader)
    metrics = compute_metrics(test_pred, test_label)

    # ROC / PR 曲线数据（用于绘制 AUROC / AUPRC 曲线）
    fpr, tpr, roc_thresholds = roc_curve(test_label, test_pred)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(test_label, test_pred)
    curve_data = {
        'roc_fpr': fpr, 'roc_tpr': tpr, 'roc_thresholds': roc_thresholds,
        'pr_precision': pr_precision, 'pr_recall': pr_recall, 'pr_thresholds': pr_thresholds,
        'y_pred': [1 if i else 0 for i in (test_pred >= 0.5)],
        'y_prob': test_pred, 'y_true': test_label,
    }
    return metrics, test_pred, test_label, curve_data

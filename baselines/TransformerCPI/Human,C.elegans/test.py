# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/5/20 20:49
@author: LiFan Chen
@Filename: test.py
@Software: PyCharm
@desc: 生成AUROC和AUPRC曲线图并保存数据
"""
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from model_glu import *
from mol_featurizer import mol_features
from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec

def load_dataset_from_csv(path, w2v_model, device):
    df = pd.read_csv(path)
    dataset = []

    for _, row in df.iterrows():
        smi = row['SMILES']
        sequence = row['Protein']
        label = row['Y']
        
        atom_feature, adj = mol_features(smi)
        protein_embedding = get_protein_embedding(w2v_model, seq_to_kmers(sequence))

        compound_feature = torch.FloatTensor(atom_feature).to(device)
        adjacency = torch.FloatTensor(adj).to(device)
        protein_feature = torch.FloatTensor(protein_embedding).to(device)
        interaction_label = torch.LongTensor([label]).to(device)

        dataset.append((compound_feature, adjacency, protein_feature, interaction_label))

    return dataset

def plot_roc_curve(fpr, tpr, auc_score, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线已保存至: {save_path}")

def plot_prc_curve(recall, precision, ap_score, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {ap_score:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall (PRC) Curve', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PRC曲线已保存至: {save_path}")

if __name__ == "__main__":
    DATASET = "biosnap"
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('使用GPU...')
    else:
        device = torch.device('cpu')
        print('使用CPU!!!')

    print("加载Word2Vec模型...")
    w2v_model = Word2Vec.load("word2vec_30.model")
    print("加载测试数据...")
    dataset_test = load_dataset_from_csv('dataset/'+DATASET+'/random/test.csv', w2v_model, device)
    print(f"测试集样本数: {len(dataset_test)}")

    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    kernel_size = 5

    encoder = Encoder(protein_dim, hid_dim, 3, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    model.to(device)

    model_path = 'dataset/'+DATASET+'/model/best_model.pt'
    print(f"加载模型: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    tester = Tester(model)

    print("生成预测结果...")
    y_true, y_scores = tester.get_predictions(dataset_test)

    print("计算ROC曲线...")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = auc(fpr, tpr)
    print(f"AUROC: {auroc:.4f}")

    print("计算PRC曲线...")
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    print(f"AUPRC (AP): {ap:.4f}")

    output_dir = 'dataset/'+DATASET+'/result'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    npy_dir = os.path.join(output_dir, 'curve_data')
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    print("保存曲线数据为npy格式...")
    np.save(os.path.join(npy_dir, 'roc_fpr.npy'), fpr)
    np.save(os.path.join(npy_dir, 'roc_tpr.npy'), tpr)
    np.save(os.path.join(npy_dir, 'prc_recall.npy'), recall)
    np.save(os.path.join(npy_dir, 'prc_precision.npy'), precision)
    np.save(os.path.join(npy_dir, 'y_true.npy'), y_true)
    np.save(os.path.join(npy_dir, 'y_scores.npy'), y_scores)
    print(f"曲线数据已保存至: {npy_dir}")

    print("绘制并保存曲线图...")
    plot_roc_curve(fpr, tpr, auroc, os.path.join(output_dir, 'ROC_curve.png'))
    plot_prc_curve(recall, precision, ap, os.path.join(output_dir, 'PRC_curve.png'))

    print("\n=== 测试结果汇总 ===")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC (Average Precision): {ap:.4f}")
    print("完成!")

import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score,
                             roc_curve, precision_recall_curve)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


def _read_csv_rows(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        idx_smiles = header.index('SMILES')
        idx_protein = header.index('Protein')
        idx_y = header.index('Y')
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split(',')
            rows.append((parts[idx_smiles], parts[idx_protein], int(float(parts[idx_y]))))
    return rows


def _split_sequence(sequence, ngram, word_dict):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def _create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def _create_ijbonddict(mol, bond_dict):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def _extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict):
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):
            fingerprints = [None] * len(nodes)
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints[i] = fingerprint_dict[fingerprint]

            nodes = fingerprints

            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    if any(v is None for v in fingerprints):
        return None
    return np.asarray(fingerprints, dtype=np.int64)


def _create_adjacency(mol):
    return np.array(Chem.GetAdjacencyMatrix(mol))


def _featurize_rows(rows, radius, ngram, atom_dict, bond_dict, fingerprint_dict, edge_dict, word_dict):
    compounds, adjacencies, proteins, interactions = [], [], [], []
    skipped = 0
    for smiles, sequence, y in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped += 1
            continue
        mol = Chem.AddHs(mol)

        atoms = _create_atoms(mol, atom_dict)
        i_jbond_dict = _create_ijbonddict(mol, bond_dict)
        fingerprints = _extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict)
        if fingerprints is None:
            skipped += 1
            continue

        adjacency = _create_adjacency(mol)
        words = _split_sequence(sequence, ngram, word_dict)

        compounds.append(fingerprints)
        adjacencies.append(adjacency)
        proteins.append(words)
        interactions.append(np.array([y], dtype=np.int64))

    return compounds, adjacencies, proteins, interactions, skipped


def get_available_datasets(dataset_base_dir):
    datasets = []
    for item in os.listdir(dataset_base_dir):
        item_path = os.path.join(dataset_base_dir, item)
        if os.path.isdir(item_path):
            random_path = os.path.join(item_path, 'random')
            if os.path.exists(random_path):
                train_file = os.path.join(random_path, 'train.csv')
                test_file = os.path.join(random_path, 'test.csv')
                if os.path.exists(train_file) and os.path.exists(test_file):
                    datasets.append(item)
    return sorted(datasets)


def build_model(n_fingerprint, n_word, dim, layer_gnn, window, layer_cnn, layer_output):
    class CompoundProteinInteractionPrediction(nn.Module):
        def __init__(self, n_fingerprint, n_word, dim, layer_gnn, window, layer_cnn, layer_output):
            super(CompoundProteinInteractionPrediction, self).__init__()
            self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
            self.embed_word = nn.Embedding(n_word, dim)
            self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
            self.W_cnn = nn.ModuleList([nn.Conv2d(
                         in_channels=1, out_channels=1, kernel_size=2*window+1,
                         stride=1, padding=window) for _ in range(layer_cnn)])
            self.W_attention = nn.Linear(dim, dim)
            self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim) for _ in range(layer_output)])
            self.W_interaction = nn.Linear(2*dim, 2)

        def gnn(self, xs, A, layer):
            for i in range(layer):
                hs = torch.relu(self.W_gnn[i](xs))
                xs = xs + torch.matmul(A, hs)
            return torch.unsqueeze(torch.mean(xs, 0), 0)

        def attention_cnn(self, x, xs, layer):
            xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
            for i in range(layer):
                xs = torch.relu(self.W_cnn[i](xs))
            xs = torch.squeeze(torch.squeeze(xs, 0), 0)

            h = torch.relu(self.W_attention(x))
            hs = torch.relu(self.W_attention(xs))
            weights = torch.tanh(F.linear(h, hs))
            ys = torch.t(weights) * hs

            return torch.unsqueeze(torch.mean(ys, 0), 0)

        def forward(self, inputs):
            fingerprints, adjacency, words = inputs

            fingerprint_vectors = self.embed_fingerprint(fingerprints)
            compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

            word_vectors = self.embed_word(words)
            protein_vector = self.attention_cnn(compound_vector, word_vectors, layer_cnn)

            cat_vector = torch.cat((compound_vector, protein_vector), 1)
            for j in range(layer_output):
                cat_vector = torch.relu(self.W_out[j](cat_vector))
            interaction = self.W_interaction(cat_vector)

            return interaction

        def predict(self, data):
            inputs, correct_interaction = data[:-1], data[-1]
            predicted_interaction = self.forward(inputs)

            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores

    return CompoundProteinInteractionPrediction(n_fingerprint, n_word, dim, layer_gnn, window, layer_cnn, layer_output)


def evaluate(model, dataset, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for data in dataset:
            correct_labels, predicted_labels, predicted_scores = model.predict(data)
            y_true.extend(list(map(int, np.ravel(correct_labels).tolist())))
            y_pred.extend(list(map(int, np.ravel(predicted_labels).tolist())))
            y_score.extend(list(map(float, np.ravel(predicted_scores).tolist())))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    roc_fpr, roc_tpr, roc_thresholds = None, None, None
    pr_precision, pr_recall, pr_thresholds = None, None, None

    metrics = {}
    if len(np.unique(y_true)) < 2:
        metrics['AUROC'] = float('nan')
        metrics['AUPRC'] = float('nan')
    else:
        metrics['AUROC'] = roc_auc_score(y_true, y_score)
        metrics['AUPRC'] = average_precision_score(y_true, y_score)
        
        roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true, y_score)
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_score)

    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics, y_true, y_pred, y_score, roc_fpr, roc_tpr, roc_thresholds, pr_precision, pr_recall, pr_thresholds


def find_model(model_dir, setting):
    model_path = os.path.join(model_dir, setting + '--best')
    if os.path.exists(model_path):
        return model_path
    model_path = os.path.join(model_dir, setting)
    if os.path.exists(model_path):
        return model_path
    return None


if __name__ == "__main__":

    dataset_base_dir = os.path.join(PROJECT_ROOT, 'dataset')
    model_base_dir = os.path.join(PROJECT_ROOT, 'output', 'model')
    output_base_dir = os.path.join(PROJECT_ROOT, 'output', 'results')

    radius = 2
    ngram = 3
    dim = 10
    layer_gnn = 3
    side = 5
    window = 2 * side + 1
    layer_cnn = 3
    layer_output = 3

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using GPU...')
    else:
        device = torch.device('cpu')
        print('Using CPU...')

    datasets = get_available_datasets(dataset_base_dir)
    print(f"Found {len(datasets)} datasets: {datasets}")

    all_results = []

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Testing on dataset: {dataset_name}")
        print('='*60)

        setting = f"{dataset_name}--radius{radius}--ngram{ngram}--dim{dim}--layer_gnn{layer_gnn}--window{window}--layer_cnn{layer_cnn}--layer_output{layer_output}--lr0.001--lr_decay0.5--decay_interval10--weight_decay1e-06--iteration60"

        model_path = find_model(model_base_dir, setting)
        if model_path is None:
            print(f"Warning: Model not found for {dataset_name}, skipping...")
            continue

        print(f"Loading model: {model_path}")

        dir_random = os.path.join(dataset_base_dir, dataset_name, 'random')
        train_rows = _read_csv_rows(os.path.join(dir_random, 'train.csv'))
        val_rows = _read_csv_rows(os.path.join(dir_random, 'val.csv'))
        test_rows = _read_csv_rows(os.path.join(dir_random, 'test.csv'))

        all_rows = train_rows + val_rows + test_rows

        atom_dict = defaultdict(lambda: len(atom_dict))
        bond_dict = defaultdict(lambda: len(bond_dict))
        fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
        edge_dict = defaultdict(lambda: len(edge_dict))
        word_dict = defaultdict(lambda: len(word_dict))

        _featurize_rows(all_rows, radius, ngram, atom_dict, bond_dict, fingerprint_dict, edge_dict, word_dict)

        n_fingerprint = len(fingerprint_dict)
        n_word = len(word_dict)

        test_compounds, test_adjacencies, test_proteins, test_interactions, test_skipped = _featurize_rows(
            test_rows, radius, ngram, atom_dict, bond_dict, fingerprint_dict, edge_dict, word_dict)

        if test_skipped > 0:
            print(f'Warning: Skipped {test_skipped} invalid SMILES in test set')

        def to_tensor_dataset(compounds, adjacencies, proteins, interactions):
            c = [torch.LongTensor(d).to(device) for d in compounds]
            a = [torch.FloatTensor(d).to(device) for d in adjacencies]
            p = [torch.LongTensor(d).to(device) for d in proteins]
            y = [torch.LongTensor(d).to(device) for d in interactions]
            return list(zip(c, a, p, y))

        dataset_test = to_tensor_dataset(test_compounds, test_adjacencies, test_proteins, test_interactions)

        torch.manual_seed(1234)
        model = build_model(n_fingerprint, n_word, dim, layer_gnn, window, layer_cnn, layer_output).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        metrics, y_true, y_pred, y_score, roc_fpr, roc_tpr, roc_thresholds, pr_precision, pr_recall, pr_thresholds = evaluate(model, dataset_test, device)

        print(f"\nTest Results for {dataset_name}:")
        print(f"  AUROC: {metrics['AUROC']:.4f}")
        print(f"  AUPRC: {metrics['AUPRC']:.4f}")
        print(f"  F1: {metrics['F1']:.4f}")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  Sensitivity: {metrics['Sensitivity']:.4f}")
        print(f"  Specificity: {metrics['Specificity']:.4f}")

        result_dir = os.path.join(output_base_dir, dataset_name)
        os.makedirs(result_dir, exist_ok=True)

        prefix = f"{dataset_name}_test"

        np.save(os.path.join(result_dir, f"{prefix}_y_true.npy"), y_true)
        np.save(os.path.join(result_dir, f"{prefix}_y_pred.npy"), y_pred)
        np.save(os.path.join(result_dir, f"{prefix}_y_score.npy"), y_score)
        
        if roc_fpr is not None:
            np.save(os.path.join(result_dir, f"{prefix}_roc_fpr.npy"), roc_fpr)
            np.save(os.path.join(result_dir, f"{prefix}_roc_tpr.npy"), roc_tpr)
            np.save(os.path.join(result_dir, f"{prefix}_roc_thresholds.npy"), roc_thresholds)
            np.save(os.path.join(result_dir, f"{prefix}_pr_precision.npy"), pr_precision)
            np.save(os.path.join(result_dir, f"{prefix}_pr_recall.npy"), pr_recall)
            np.save(os.path.join(result_dir, f"{prefix}_pr_thresholds.npy"), pr_thresholds)

        print(f"\nSaved prediction files to {result_dir}:")
        print(f"  {prefix}_y_true.npy")
        print(f"  {prefix}_y_pred.npy")
        print(f"  {prefix}_y_score.npy")
        print(f"  {prefix}_roc_fpr.npy")
        print(f"  {prefix}_roc_tpr.npy")
        print(f"  {prefix}_roc_thresholds.npy")
        print(f"  {prefix}_pr_precision.npy")
        print(f"  {prefix}_pr_recall.npy")
        print(f"  {prefix}_pr_thresholds.npy")

        all_results.append({
            'dataset': dataset_name,
            'auroc': metrics['AUROC'],
            'auprc': metrics['AUPRC'],
            'f1': metrics['F1'],
            'accuracy': metrics['Accuracy'],
            'sensitivity': metrics['Sensitivity'],
            'specificity': metrics['Specificity']
        })

    print(f"\n{'='*60}")
    print("Summary of Test Results")
    print('='*60)
    print(f"{'Dataset':<15} {'AUROC':<8} {'AUPRC':<8} {'F1':<8} {'Acc':<8}")
    print('-'*60)
    for result in all_results:
        print(f"{result['dataset']:<15} {result['auroc']:<8.4f} {result['auprc']:<8.4f} {result['f1']:<8.4f} {result['accuracy']:<8.4f}")

    summary_path = os.path.join(output_base_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Dataset\tAUROC\tAUPRC\tF1\tAccuracy\tSensitivity\tSpecificity\n")
        for result in all_results:
            f.write(f"{result['dataset']}\t{result['auroc']:.4f}\t{result['auprc']:.4f}\t{result['f1']:.4f}\t{result['accuracy']:.4f}\t{result['sensitivity']:.4f}\t{result['specificity']:.4f}\n")
    print(f"\nSummary saved to: {summary_path}")

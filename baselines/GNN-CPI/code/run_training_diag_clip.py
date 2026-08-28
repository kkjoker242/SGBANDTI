import sys
import timeit
import pickle
import os
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rdkit import Chem

from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_ARGS = {
    'DATASET': 'bindingdb',
    'radius': 2,
    'ngram': 3,
    'dim': 10,
    'layer_gnn': 3,
    'side': 5,
    'layer_cnn': 3,
    'layer_output': 3,
    'lr': 1e-3,
    'lr_decay': 0.5,
    'decay_interval': 10,
    'weight_decay': 1e-6,
    'iteration': 60,
    'seeds': [42, 52, 62, 72, 82],
}


def parse_seed_values(seed_arg):
    if isinstance(seed_arg, (list, tuple)):
        return [int(seed) for seed in seed_arg]

    seeds = []
    for part in str(seed_arg).split(','):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError('At least one seed must be provided.')
    return seeds


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 2)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

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

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)

        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def evaluate(self, dataset):
        y_true, y_pred, y_score = [], [], []
        for data in dataset:
            (correct_labels, predicted_labels,
             predicted_scores) = self.model(data, train=False)
            y_true.extend(list(map(int, np.ravel(correct_labels).tolist())))
            y_pred.extend(list(map(int, np.ravel(predicted_labels).tolist())))
            y_score.extend(list(map(float, np.ravel(predicted_scores).tolist())))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)

        metrics = {}
        if len(np.unique(y_true)) < 2:
            metrics['AUROC'] = float('nan')
            metrics['AUPRC'] = float('nan')
        else:
            metrics['AUROC'] = roc_auc_score(y_true, y_score)
            metrics['AUPRC'] = average_precision_score(y_true, y_score)

        metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return metrics

    def append_metrics_row(self, values, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, values)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


if __name__ == "__main__":

    """Hyperparameters."""
    if len(sys.argv) == 1:
        DATASET = DEFAULT_ARGS['DATASET']
        radius = DEFAULT_ARGS['radius']
        ngram = DEFAULT_ARGS['ngram']
        dim = DEFAULT_ARGS['dim']
        layer_gnn = DEFAULT_ARGS['layer_gnn']
        side = DEFAULT_ARGS['side']
        window = 2 * side + 1
        layer_cnn = DEFAULT_ARGS['layer_cnn']
        layer_output = DEFAULT_ARGS['layer_output']
        lr = DEFAULT_ARGS['lr']
        lr_decay = DEFAULT_ARGS['lr_decay']
        decay_interval = DEFAULT_ARGS['decay_interval']
        weight_decay = DEFAULT_ARGS['weight_decay']
        iteration = DEFAULT_ARGS['iteration']
        seeds = list(DEFAULT_ARGS['seeds'])
        setting = (
            f'{DATASET}--radius{radius}--ngram{ngram}--dim{dim}'
            f'--layer_gnn{layer_gnn}--window{window}--layer_cnn{layer_cnn}'
            f'--layer_output{layer_output}--lr{lr}--lr_decay{lr_decay}'
            f'--decay_interval{decay_interval}--weight_decay{weight_decay}'
            f'--iteration{iteration}'
        )
        print('No CLI args provided. Using default configuration:')
        print(setting)
        print('Seeds:', ','.join(map(str, seeds)))
    else:
        args = sys.argv[1:]
        if len(args) not in (14, 15):
            raise ValueError(
                'Expected 14 or 15 arguments: '
                'DATASET radius ngram dim layer_gnn window layer_cnn layer_output '
                'lr lr_decay decay_interval weight_decay iteration setting [seeds]'
            )

        (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
         lr, lr_decay, decay_interval, weight_decay, iteration,
         setting) = args[:14]
        (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
         iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
                                decay_interval, iteration])
        lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])
        radius = int(radius)
        ngram = int(ngram)
        if len(args) == 15:
            seeds = parse_seed_values(args[14])
        else:
            seeds = [1234]

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

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
        """Weisfeiler-Lehman style fingerprints (same logic as preprocess_data.py)."""
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

    """Load fixed train/val/test splits from CSV."""
    dir_random = os.path.join(PROJECT_ROOT, 'dataset', DATASET, 'random')
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

    train_compounds, train_adjacencies, train_proteins, train_interactions, train_skipped = _featurize_rows(
        train_rows, radius, ngram, atom_dict, bond_dict, fingerprint_dict, edge_dict, word_dict)
    val_compounds, val_adjacencies, val_proteins, val_interactions, val_skipped = _featurize_rows(
        val_rows, radius, ngram, atom_dict, bond_dict, fingerprint_dict, edge_dict, word_dict)
    test_compounds, test_adjacencies, test_proteins, test_interactions, test_skipped = _featurize_rows(
        test_rows, radius, ngram, atom_dict, bond_dict, fingerprint_dict, edge_dict, word_dict)

    if (train_skipped + val_skipped + test_skipped) > 0:
        print('Skipped invalid SMILES rows:', train_skipped + val_skipped + test_skipped)

    def _to_dataset(compounds, adjacencies, proteins, interactions):
        c = [torch.LongTensor(d).to(device) for d in compounds]
        a = [torch.FloatTensor(d).to(device) for d in adjacencies]
        p = [torch.LongTensor(d).to(device) for d in proteins]
        y = [torch.LongTensor(d).to(device) for d in interactions]
        return list(zip(c, a, p, y))

    dataset_train = _to_dataset(train_compounds, train_adjacencies, train_proteins, train_interactions)
    dataset_dev = _to_dataset(val_compounds, val_adjacencies, val_proteins, val_interactions)
    dataset_test = _to_dataset(test_compounds, test_adjacencies, test_proteins, test_interactions)

    """Output files."""
    result_dir = os.path.join(PROJECT_ROOT, 'output', 'result')
    model_dir = os.path.join(PROJECT_ROOT, 'output', 'model')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    header = (
        'Epoch\tTime(sec)\tLoss_train\t'
        'val_AUROC\tval_AUPRC\tval_F1\tval_Sensitivity\tval_Specificity\tval_Accuracy\t'
        'test_AUROC\ttest_AUPRC\ttest_F1\ttest_Sensitivity\ttest_Specificity\ttest_Accuracy'
    )
    summary_file = os.path.join(result_dir, 'Summary--' + setting + '.txt')
    summary_keys = ['AUROC', 'AUPRC', 'F1', 'Sensitivity', 'Specificity', 'Accuracy']
    seed_summaries = []

    for seed in seeds:
        set_random_seed(seed)

        model = CompoundProteinInteractionPrediction().to(device)
        trainer = Trainer(model)
        tester = Tester(model)

        seed_setting = setting + '--seed' + str(seed)
        file_metrics = os.path.join(result_dir, 'Metrics--' + seed_setting + '.txt')
        file_model_last = os.path.join(model_dir, seed_setting)
        file_model_best = os.path.join(model_dir, seed_setting + '--best')

        with open(file_metrics, 'w') as f:
            f.write(header + '\n')

        print('Training...')
        print('Seed:', seed)
        print(header)

        best_val_auprc = -1.0
        best_epoch = -1
        best_test_metrics = None

        start = timeit.default_timer()

        for epoch in range(1, iteration):

            if epoch % decay_interval == 0:
                trainer.optimizer.param_groups[0]['lr'] *= lr_decay

            loss_train = trainer.train(dataset_train)

            val_metrics = tester.evaluate(dataset_dev)
            test_metrics = tester.evaluate(dataset_test)

            end = timeit.default_timer()
            time = end - start

            row = [
                epoch, time, loss_train,
                val_metrics['AUROC'], val_metrics['AUPRC'], val_metrics['F1'],
                val_metrics['Sensitivity'], val_metrics['Specificity'], val_metrics['Accuracy'],
                test_metrics['AUROC'], test_metrics['AUPRC'], test_metrics['F1'],
                test_metrics['Sensitivity'], test_metrics['Specificity'], test_metrics['Accuracy'],
            ]
            tester.append_metrics_row(row, file_metrics)
            tester.save_model(model, file_model_last)

            if val_metrics['AUPRC'] > best_val_auprc:
                best_val_auprc = val_metrics['AUPRC']
                best_epoch = epoch
                best_test_metrics = dict(test_metrics)
                tester.save_model(model, file_model_best)

            print('\t'.join(map(str, row)))

        with open(file_metrics, 'a') as f:
            f.write('BEST_EPOCH\t' + str(best_epoch) + '\n')
            f.write('BEST_VAL_AUPRC\t' + str(best_val_auprc) + '\n')
            if best_test_metrics is not None:
                f.write('BEST_TEST\t' + '\t'.join([
                    str(best_test_metrics['AUROC']),
                    str(best_test_metrics['AUPRC']),
                    str(best_test_metrics['F1']),
                    str(best_test_metrics['Sensitivity']),
                    str(best_test_metrics['Specificity']),
                    str(best_test_metrics['Accuracy']),
                ]) + '\n')

        if best_test_metrics is not None:
            seed_summaries.append({
                'seed': seed,
                'best_epoch': best_epoch,
                'best_val_auprc': best_val_auprc,
                'best_test_metrics': best_test_metrics,
            })

    if seed_summaries:
        with open(summary_file, 'w') as f:
            f.write('seed\tbest_epoch\tbest_val_auprc\t' + '\t'.join(summary_keys) + '\n')
            for item in seed_summaries:
                f.write(
                    str(item['seed']) + '\t' +
                    str(item['best_epoch']) + '\t' +
                    str(item['best_val_auprc']) + '\t' +
                    '\t'.join(str(item['best_test_metrics'][key]) for key in summary_keys) + '\n'
                )

            mean_values = {
                key: float(np.mean([item['best_test_metrics'][key] for item in seed_summaries]))
                for key in summary_keys
            }
            std_values = {
                key: float(np.std([item['best_test_metrics'][key] for item in seed_summaries]))
                for key in summary_keys
            }

            f.write(
                'MEAN\t-\t' +
                str(float(np.mean([item['best_val_auprc'] for item in seed_summaries]))) + '\t' +
                '\t'.join(str(mean_values[key]) for key in summary_keys) + '\n'
            )
            f.write(
                'STD\t-\t' +
                str(float(np.std([item['best_val_auprc'] for item in seed_summaries]))) + '\t' +
                '\t'.join(str(std_values[key]) for key in summary_keys) + '\n'
            )

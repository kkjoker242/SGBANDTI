# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/5/20 20:49
@author: LiFan Chen
@Filename: main_glu.py
@Software: PyCharm
"""
import torch
import numpy as np
import pandas as pd
import random
import os
import time
from model_glu import *
from mol_featurizer import mol_features
from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec
import timeit


def load_dataset_from_csv(path, w2v_model, device):
    df = pd.read_csv(path)
    dataset = []

    for _, row in df.iterrows():
        smi = row['SMILES']
        sequence = row['Protein']
        label = row['Y']

        atom_feature, adj = mol_features(smi)
        protein_embedding = get_protein_embedding(w2v_model, seq_to_kmers(sequence))

        # Convert to tensors
        compound_feature = torch.FloatTensor(atom_feature).to(device)
        adjacency = torch.FloatTensor(adj).to(device)
        protein_feature = torch.FloatTensor(protein_embedding).to(device)
        interaction_label = torch.LongTensor([label]).to(device)

        dataset.append((compound_feature, adjacency, protein_feature, interaction_label))

    return dataset


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def run_experiment(seed, dataset_name, w2v_model, device):
    set_random_seed(seed)

    print(f"\nRunning experiment with seed {seed}...")
    print("Loading data from CSV files...")
    dataset_train = load_dataset_from_csv('dataset/' + dataset_name + '/random/train.csv', w2v_model, device)
    dataset_dev = load_dataset_from_csv('dataset/' + dataset_name + '/random/val.csv', w2v_model, device)
    dataset_test = load_dataset_from_csv('dataset/' + dataset_name + '/random/test.csv', w2v_model, device)
    print("Data loaded.")

    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 64
    lr = 1e-3
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 0.5
    iteration = 20
    kernel_size = 5

    encoder = Encoder(protein_dim, hid_dim, 3, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)

    result_dir = 'dataset/' + dataset_name + '/result'
    model_dir = 'dataset/' + dataset_name + '/model'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_AUCs = result_dir + f'/best_metrics_seed_{seed}.txt'
    file_model = model_dir + f'/best_model_seed_{seed}.pt'
    results_header = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tAUROC_test\tAUPRC_test\tF1_test\tSensitivity_test\tSpecificity_test\tAccuracy_test')
    with open(file_AUCs, 'w') as f:
        f.write(results_header + '\n')

    print('Training...')
    print(results_header)
    start = timeit.default_timer()

    max_AUC_dev = 0
    epoch_label = 0
    best_test_metrics = None
    for epoch in range(1, iteration + 1):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train, device)
        AUC_dev, _, _ = tester.test(dataset_dev)

        end = timeit.default_timer()
        elapsed_time = end - start

        if AUC_dev > max_AUC_dev:
            max_AUC_dev = AUC_dev
            test_metrics = tester.test(dataset_test, full_metrics=True)
            best_test_metrics = [epoch, round(elapsed_time, 2), round(loss_train, 4), round(AUC_dev, 4)] + [round(m, 4) for m in test_metrics]
            tester.save_model(model, file_model)
            epoch_label = epoch
            print('\t'.join(map(str, best_test_metrics)))
            with open(file_AUCs, 'w') as f:
                f.write(results_header + '\n')
                f.write('\t'.join(map(str, best_test_metrics)) + '\n')

    print("\nThe best model was found at epoch", epoch_label)
    return best_test_metrics


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument('--data', default='biosnap', choices=['biosnap', 'bindingdb'])
    _ap.add_argument('--seeds', default=None, help='comma-separated seeds, e.g. 62')
    _args = _ap.parse_args()

    SEEDS = [int(s) for s in _args.seeds.split(',')] if _args.seeds else [42, 52, 62, 72, 82]
    DATASET = _args.data

    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    print("Loading Word2Vec model...")
    w2v_model = Word2Vec.load("word2vec_30.model")

    all_results = {}
    for seed in SEEDS:
        all_results[seed] = run_experiment(seed, DATASET, w2v_model, device)

    print("\nFinished all seed runs.")
    for seed, metrics in all_results.items():
        if metrics is not None:
            print(f"Seed {seed}:" + '\t' + '\t'.join(map(str, metrics)))

import copy
import os
from time import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
from torch import nn
from torch.autograd import Variable
from torch.utils import data

from argparse import ArgumentParser
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = ArgumentParser(description='MolTrans Training.')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--task', choices=['biosnap', 'bindingdb', 'GPCR' , 'human', 'C.elegans'],
                    default='bindingdb', type=str, metavar='TASK',
                    help='Task name. Could be biosnap, bindingdb and davis.')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
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


def test(data_generator, model):
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

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy : ', accuracy1)

    # sklearn confusion_matrix layout: [[TN, FP], [FN, TP]]
    # (fixed: original code had sensitivity/specificity swapped)
    sensitivity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,
                                                                                              outputs), y_pred, loss.item(), sensitivity1, specificity1, accuracy1


def main(seed=2):
    config = BIN_config_DBPE()
    args = parser.parse_args()
    config['batch_size'] = args.batch_size

    torch.manual_seed(seed)
    np.random.seed(seed + 1)  # 使用不同的种子偏移以保证多样性

    loss_history = []

    model = BIN_Interaction_Flat(**config)

    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim=0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('--- Data Preparation ---')
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.workers,
              'drop_last': True}

    dataFolder = get_task(args.task)

    df_train = pd.read_csv(dataFolder + '/train.csv')
    df_train.columns = [col.strip() for col in df_train.columns]
    df_val = pd.read_csv(dataFolder + '/val.csv')
    df_val.columns = [col.strip() for col in df_val.columns]
    df_test = pd.read_csv(dataFolder + '/test.csv')
    df_test.columns = [col.strip() for col in df_test.columns]

    training_set = BIN_Data_Encoder(df_train.index.values, df_train['Label'].values, df_train)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = BIN_Data_Encoder(df_val.index.values, df_val['Label'].values, df_val)
    validation_generator = data.DataLoader(validation_set, **params)

    testing_set = BIN_Data_Encoder(df_test.index.values, df_test['Label'].values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)

    with torch.set_grad_enabled(False):
        auc, auprc, f1, logits, loss, sens, spec, acc = test(testing_generator, model_max)
        print('Initial Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(
            f1) + ' , Test loss: ' + str(loss))

    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in range(args.epochs):
        model.train()
        for i, (d, p, d_mask, p_mask, label) in enumerate(training_generator):
            score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())

            label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            n = torch.squeeze(m(score))

            loss = loss_fct(n, label)
            # 存标量而非 GPU 张量，避免 loss_history 累积占满显存（原代码 OOM 原因）
            loss_history.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i % 1000 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))

        # every epoch test
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss, sens, spec, acc = test(validation_generator, model)
            if auc > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = auc
            print('Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc) + ' , AUPRC: ' + str(
                auprc) + ' , F1: ' + str(f1))

    print('--- Go for Testing ---')
    os.makedirs('./model_checkpoints', exist_ok=True)
    model_save_path = os.path.join('./model_checkpoints', f'best_model_{args.task}_seed{seed}.pt')
    torch.save(model_max.state_dict(), model_save_path)
    print(f'Model saved to: {model_save_path}')

    test_result = {}
    try:
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss, sens, spec, acc = test(testing_generator, model_max)
            print(
                'Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) + ' , Test loss: ' + str(
                    loss) + ' , Sensitivity: ' + str(sens) + ' , Specificity: ' + str(spec) + ' , Accuracy: ' + str(acc))
            test_result = {
                'seed': seed,
                'auroc': float(auc),
                'auprc': float(auprc),
                'f1': float(f1),
                'loss': float(loss),
                'sensitivity': float(sens),
                'specificity': float(spec),
                'accuracy': float(acc),
                'y_pred': np.array(logits)
            }
    except:
        print('testing failed')
        test_result = {
            'seed': seed,
            'auroc': 0.0,
            'auprc': 0.0,
            'f1': 0.0,
            'loss': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'accuracy': 0.0,
            'y_pred': np.array([])
        }
    return model_max, loss_history, test_result


s = time()
args = parser.parse_args()
all_results = []
all_test_results = []
for seed in args.seeds:
    print(f'\n{"="*60}')
    print(f'>>> Training with seed = {seed} <<<')
    print(f'{"="*60}\n')
    model_max, loss_history, test_result = main(seed=seed)
    all_results.append((seed, model_max, loss_history))
    all_test_results.append(test_result)
e = time()
print(f'\nTotal training time for {len(args.seeds)} seeds: {e - s:.2f}s')

print(f'\n{"="*60}')
print(f'>>> Final Test Results Summary across {len(args.seeds)} seeds <<<')
print(f'{"="*60}')

results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

aurocs = [r['auroc'] for r in all_test_results]
auprcs = [r['auprc'] for r in all_test_results]
f1s = [r['f1'] for r in all_test_results]
losses = [r['loss'] for r in all_test_results]
senss = [r['sensitivity'] for r in all_test_results]
specs = [r['specificity'] for r in all_test_results]
accs = [r['accuracy'] for r in all_test_results]

print(f'AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}')
print(f'AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}')
print(f'F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
print(f'Loss:  {np.mean(losses):.4f} ± {np.std(losses):.4f}')
print(f'Sens:  {np.mean(senss):.4f} ± {np.std(senss):.4f}')
print(f'Spec:  {np.mean(specs):.4f} ± {np.std(specs):.4f}')
print(f'Acc:   {np.mean(accs):.4f} ± {np.std(accs):.4f}')

print(f'\nDetailed per-seed results:')
print(f'{"Seed":<8}{"AUROC":<12}{"AUPRC":<12}{"F1":<12}{"Sens":<12}{"Spec":<12}{"Acc":<12}{"Loss":<12}')
print('-' * 92)
for r in all_test_results:
    print(f'{r["seed"]:<8}{r["auroc"]:<12.4f}{r["auprc"]:<12.4f}{r["f1"]:<12.4f}{r["sensitivity"]:<12.4f}{r["specificity"]:<12.4f}{r["accuracy"]:<12.4f}{r["loss"]:<12.4f}')

summary_file = os.path.join(results_dir, f'train_test_results_{args.task}.txt')
with open(summary_file, 'w') as f:
    f.write(f'Test Results Summary for {args.task.upper()}\n')
    f.write(f'Number of seeds: {len(args.seeds)}\n')
    f.write(f'Seeds: {args.seeds}\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}\n')
    f.write(f'AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}\n')
    f.write(f'F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}\n')
    f.write(f'Loss:  {np.mean(losses):.4f} ± {np.std(losses):.4f}\n')
    f.write(f'Sens:  {np.mean(senss):.4f} ± {np.std(senss):.4f}\n')
    f.write(f'Spec:  {np.mean(specs):.4f} ± {np.std(specs):.4f}\n')
    f.write(f'Acc:   {np.mean(accs):.4f} ± {np.std(accs):.4f}\n\n')
    f.write(f'Detailed per-seed results:\n')
    f.write(f'{"Seed":<8}{"AUROC":<12}{"AUPRC":<12}{"F1":<12}{"Sens":<12}{"Spec":<12}{"Acc":<12}{"Loss":<12}\n')
    f.write('-' * 92 + '\n')
    for r in all_test_results:
        f.write(f'{r["seed"]:<8}{r["auroc"]:<12.4f}{r["auprc"]:<12.4f}{r["f1"]:<12.4f}{r["sensitivity"]:<12.4f}{r["specificity"]:<12.4f}{r["accuracy"]:<12.4f}{r["loss"]:<12.4f}\n')
print(f'\nResults summary saved to: {summary_file}')

np.save(os.path.join(results_dir, f'aurocs_{args.task}.npy'), np.array(aurocs))
np.save(os.path.join(results_dir, f'auprcs_{args.task}.npy'), np.array(auprcs))
np.save(os.path.join(results_dir, f'f1s_{args.task}.npy'), np.array(f1s))
np.save(os.path.join(results_dir, f'losses_{args.task}.npy'), np.array(losses))
np.save(os.path.join(results_dir, f'senss_{args.task}.npy'), np.array(senss))
np.save(os.path.join(results_dir, f'specs_{args.task}.npy'), np.array(specs))
np.save(os.path.join(results_dir, f'accs_{args.task}.npy'), np.array(accs))

for r in all_test_results:
    if len(r['y_pred']) > 0:
        np.save(os.path.join(results_dir, f'y_pred_{args.task}_seed{r["seed"]}.npy'), r['y_pred'])

print(f'\nNPY data files saved to: {results_dir}')

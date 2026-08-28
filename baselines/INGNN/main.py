import os
os.environ['CUDA_VISIBLE_DEVICES']="0"

import numpy as np
import pandas as pd

import espf_cnn
from utils import _load_espf

# 加载 ESPF 词表（BPE 编码所需）
_load_espf()

seeds = [42, 52, 62, 72, 82]
dataFolder = ['datasets/bindingdb/random/']
data_names = ['bindingdb_random']

metrics = ['AUROC', 'AUPRC', 'F1', 'Sensitivity', 'Specificity', 'Accuracy', 'Precision']

all_results = {}

for j in range(len(dataFolder)):
    dataset_name = data_names[j]
    all_results[dataset_name] = []

    train_df = pd.read_csv(dataFolder[j] + 'train.csv')
    val_df = pd.read_csv(dataFolder[j] + 'val.csv')
    test_df = pd.read_csv(dataFolder[j] + 'test.csv')

    print(f'\n=== Dataset: {dataset_name} ===')

    seed_results = []

    for seed in seeds:
        print(f'\n--- Seed: {seed} ---')

        metrics_result, _, _, curve_data = espf_cnn.train_espf_cnn(
            train_df, val_df, test_df, seed=seed,
        )
        auc, auprc, f1, sensitivity, specificity, acc, precision = metrics_result

        seed_results.append([auc, auprc, f1, sensitivity, specificity, acc, precision])

        print(f'Seed {seed} results: AUROC={auc:.4f}, AUPRC={auprc:.4f}, F1={f1:.4f}, '
              f'Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}, '
              f'Accuracy={acc:.4f}, Precision={precision:.4f}')

        # 保存 AUROC / AUPRC 曲线数据（参考 RF_results_bindingdb 文件结构）
        save_dir = f'results_{dataset_name}'
        os.makedirs(save_dir, exist_ok=True)
        for key, value in curve_data.items():
            np.save(os.path.join(save_dir, f'{key}.npy'), np.asarray(value))
        print(f'Curve data saved to {save_dir}/')

    all_results[dataset_name].append({'seed_results': seed_results})

print('\n\n=== Final Results Summary ===')
for dataset_name, dataset_results in all_results.items():
    for dr in dataset_results:
        seed_results = np.array(dr['seed_results'])

        means = np.mean(seed_results, axis=0)
        stds = np.std(seed_results, axis=0)

        print(f'\nDataset: {dataset_name}')
        for i, metric in enumerate(metrics):
            print(f'{metric}: {means[i]:.4f} ± {stds[i]:.4f}')

        result_df = pd.DataFrame({
            'Metric': metrics,
            'Mean': means,
            'Std': stds
        })
        result_df.to_csv(f'{dataset_name}_results_summary.csv', index=False)
        print(f'Results saved to {dataset_name}_results_summary.csv')
import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import joblib
import os
import itertools
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return np.zeros(n_bits)

def protein_to_features(protein, k_mer_sizes=[1, 2, 3]):
    """Convert protein sequence to features using amino acid composition and k-mer features"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    
    features = []
    
    for aa in amino_acids:
        features.append(protein.count(aa) / len(protein) if len(protein) > 0 else 0)
    
    for k in k_mer_sizes:
        k_mer_counts = Counter([protein[i:i+k] for i in range(len(protein)-k+1)])
        total = sum(k_mer_counts.values())
        for km in [''.join(p) for p in itertools.product(amino_acids, repeat=k)]:
            features.append(k_mer_counts.get(km, 0) / total if total > 0 else 0)
    
    features.append(len(protein))
    
    return np.array(features)

def extract_features(df):
    """Extract features from SMILES and protein sequences"""
    drug_features = []
    protein_features = []
    
    for idx, row in df.iterrows():
        drug_fp = smiles_to_fingerprint(row['SMILES'])
        protein_feat = protein_to_features(row['Protein'])
        
        drug_features.append(drug_fp)
        protein_features.append(protein_feat)
    
    drug_features = np.array(drug_features)
    protein_features = np.array(protein_features)
    
    X = np.hstack([drug_features, protein_features])
    y = df['Y'].values
    
    return X, y

def select_threshold_f1(y_true, y_prob):
    """在验证集上按标准 F1 选择最优分类阈值（与 SGBANDTI 协议一致，避免测试集泄漏）。"""
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    # precision_recall_curve 末尾有哨兵点（threshold=0），与 thresholds 不对齐，剔除
    best_idx = int(np.argmax(f1s[:-1]))
    return float(thresholds[best_idx])

def calculate_metrics(y_true, y_prob, threshold):
    """Calculate 6 evaluation metrics and save curve data（阈值依赖指标用给定阈值）"""
    from sklearn.metrics import roc_curve, precision_recall_curve

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

    metrics = {
        'AUC': auc,
        'AUPRC': auprc,
        'F1': f1,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Accuracy': accuracy,
        'threshold': threshold,
        'roc_curve': {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': roc_thresholds
        },
        'pr_curve': {
            'precision': precision,
            'recall': recall,
            'thresholds': pr_thresholds
        },
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': y_pred
    }
    
    return metrics

def train_and_evaluate(dataset_path, dataset_name, seed):
    """Train Random Forest model and evaluate on test set（每 seed 独立，阈值=验证集 F1 最优）"""
    print(f"\n{'='*60}")
    print(f"Training on {dataset_name} dataset, seed {seed}")
    print(f"{'='*60}")

    train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    val_df = pd.read_csv(os.path.join(dataset_path, 'val.csv'))
    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    print("\nExtracting features...")
    X_train, y_train = extract_features(train_df)
    X_val, y_val = extract_features(val_df)
    X_test, y_test = extract_features(test_df)

    print(f"Feature dimension: {X_train.shape[1]}")

    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_prob)
    val_thr = select_threshold_f1(y_val, val_prob)

    print(f"\nValidation AUC: {val_auc:.4f}  Val-F1-optimal threshold: {val_thr:.4f}")

    print("\nEvaluating on test set...")
    test_prob = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, test_prob, val_thr)

    output_dir = os.path.join(f'results_{dataset_name.lower()}', f'seed_{seed}')
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'roc_fpr.npy'), metrics['roc_curve']['fpr'])
    np.save(os.path.join(output_dir, 'roc_tpr.npy'), metrics['roc_curve']['tpr'])
    np.save(os.path.join(output_dir, 'roc_thresholds.npy'), metrics['roc_curve']['thresholds'])

    np.save(os.path.join(output_dir, 'pr_precision.npy'), metrics['pr_curve']['precision'])
    np.save(os.path.join(output_dir, 'pr_recall.npy'), metrics['pr_curve']['recall'])
    np.save(os.path.join(output_dir, 'pr_thresholds.npy'), metrics['pr_curve']['thresholds'])

    np.save(os.path.join(output_dir, 'y_true.npy'), metrics['y_true'])
    np.save(os.path.join(output_dir, 'y_prob.npy'), metrics['y_prob'])
    np.save(os.path.join(output_dir, 'y_pred.npy'), metrics['y_pred'])

    print(f"\nCurve data saved to {output_dir}/")
    
    print(f"\n{'='*60}")
    print(f"Test Set Results for {dataset_name}")
    print(f"{'='*60}")
    for metric_name, value in metrics.items():
        if metric_name not in ['roc_curve', 'pr_curve', 'y_true', 'y_prob', 'y_pred']:
            print(f"{metric_name}: {value:.4f}")
    
    return metrics, model

def main():
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else 'biosnap'
    dataset_path = f'dataset/{dataset_arg}/random'
    dataset_name = 'BioSnap' if dataset_arg == 'biosnap' else 'BindingDB'
    seeds = [42, 52, 62, 72, 82]

    all_metrics = []

    if not os.path.exists(dataset_path):
        print(f"\nDataset not found: {dataset_path}")
        return

    for seed in seeds:
        metrics, model = train_and_evaluate(dataset_path, dataset_name, seed)
        metrics['seed'] = seed
        all_metrics.append(metrics)

        model_path = f'best_model_{dataset_name.lower()}_seed{seed}.joblib'
        joblib.dump(model, model_path)
        print(f"\nBest model saved to {model_path}")

    metric_cols = ['seed', 'AUC', 'AUPRC', 'F1', 'Sensitivity', 'Specificity', 'Accuracy', 'threshold']
    summary_df = pd.DataFrame([{c: m[c] for c in metric_cols} for m in all_metrics])
    output_dir = f'results_{dataset_name.lower()}'
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(output_dir, 'seed_summary.csv'), index=False)

    stats_df = summary_df[metric_cols[1:]].agg(['mean', 'std']).T
    stats_df.to_csv(os.path.join(output_dir, 'seed_summary_stats.csv'))

    print(f"\n{'='*60}")
    print("Seed summary:")
    print(summary_df.to_string(index=False))
    print("\nMean ± std:")
    for col in metric_cols[1:]:
        print(f"  {col}: {stats_df.loc[col, 'mean']:.4f} ± {stats_df.loc[col, 'std']:.4f}")

if __name__ == '__main__':
    main()

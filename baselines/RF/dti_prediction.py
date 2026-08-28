import pandas as pd
import numpy as np
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

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate 6 evaluation metrics and save curve data"""
    from sklearn.metrics import roc_curve, precision_recall_curve
    
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

def train_and_evaluate(dataset_path, dataset_name):
    """Train Random Forest model and evaluate on test set"""
    print(f"\n{'='*60}")
    print(f"Training on {dataset_name} dataset")
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
    
    best_val_auc = 0
    best_model = None
    
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=62,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    val_prob = model.predict_proba(X_val)[:, 1]
    val_pred = model.predict(X_val)
    val_auc = roc_auc_score(y_val, val_prob)
    
    print(f"\nValidation AUC: {val_auc:.4f}")
    
    best_val_auc = val_auc
    best_model = model
    
    print("\nEvaluating on test set...")
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_pred = best_model.predict(X_test)
    
    metrics = calculate_metrics(y_test, test_pred, test_prob)
    
    output_dir = f'results_{dataset_name.lower()}'
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
    
    return metrics, best_model

def main():
    datasets = [
        ('dataset/bindingdb/random', 'BindingDB'),
    ]
    
    all_results = {}
    
    for dataset_path, dataset_name in datasets:
        if os.path.exists(dataset_path):
            metrics, model = train_and_evaluate(dataset_path, dataset_name)
            all_results[dataset_name] = metrics
            
            model_path = f'best_model_{dataset_name.lower()}.joblib'
            joblib.dump(model, model_path)
            print(f"\nBest model saved to {model_path}")
        else:
            print(f"\nDataset not found: {dataset_path}")
    
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")
    for dataset_name, metrics in all_results.items():
        print(f"\n{dataset_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

if __name__ == '__main__':
    main()

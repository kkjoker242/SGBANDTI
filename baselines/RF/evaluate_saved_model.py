import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve
import joblib
import os

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
    
    features = []
    
    for aa in amino_acids:
        features.append(protein.count(aa) / len(protein) if len(protein) > 0 else 0)
    
    import itertools
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
    """Calculate 6 evaluation metrics and return curve data"""
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

def evaluate_saved_model(model_path, test_csv_path, dataset_name='test'):
    """Load saved model and evaluate on test set"""
    print(f"\n{'='*60}")
    print(f"Evaluating saved model: {model_path}")
    print(f"{'='*60}")
    
    print(f"\nLoading model from: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully!")
    
    print(f"\nLoading test data from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    print(f"Test set size: {len(test_df)}")
    
    print("\nExtracting features...")
    X_test, y_test = extract_features(test_df)
    print(f"Feature dimension: {X_test.shape[1]}")
    
    print("\nMaking predictions...")
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test, test_pred, test_prob)
    
    output_dir = f'evaluation_results_{dataset_name.lower()}'
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
    print(f"Test Set Evaluation Results")
    print(f"{'='*60}")
    for metric_name in ['AUC', 'AUPRC', 'F1', 'Sensitivity', 'Specificity', 'Accuracy']:
        print(f"{metric_name}: {metrics[metric_name]:.4f}")
    
    return metrics

def main():
    model_configs = [
        {
            'model_path': 'best_model_bindingdb.joblib',
            'test_csv_path': 'dataset/bindingdb/random/test.csv',
            'dataset_name': 'BindingDB'
        },
        {
            'model_path': 'best_model_biosnap.joblib',
            'test_csv_path': 'dataset/biosnap/random/test.csv',
            'dataset_name': 'BioSnap'
        }
    ]
    
    all_results = {}
    
    for config in model_configs:
        if os.path.exists(config['model_path']) and os.path.exists(config['test_csv_path']):
            metrics = evaluate_saved_model(
                config['model_path'],
                config['test_csv_path'],
                config['dataset_name']
            )
            all_results[config['dataset_name']] = metrics
        else:
            print(f"\nWarning: Model or test data not found:")
            print(f"  Model: {config['model_path']}")
            print(f"  Test data: {config['test_csv_path']}")
    
    if all_results:
        print(f"\n{'='*60}")
        print("Final Evaluation Summary")
        print(f"{'='*60}")
        for dataset_name, metrics in all_results.items():
            print(f"\n{dataset_name}:")
            for metric_name in ['AUC', 'AUPRC', 'F1', 'Sensitivity', 'Specificity', 'Accuracy']:
                print(f"  {metric_name}: {metrics[metric_name]:.4f}")

if __name__ == '__main__':
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from collections import Counter
    
    main()

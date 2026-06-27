# SGBANDTI

This project is for drug-target interaction (DTI) prediction with the `NGNN_BAN` model.
It is built on PyTorch, DGL, DGL-LifeSci, and PyG.

Supported workflow:

- build NestedGNN subgraph cache
- train with 5 random seeds
- evaluate a saved checkpoint on the test set

## 1. Project Files

Main files:

- `main.py`: training entry, runs 5 seeds by default
- `test.py`: evaluate a trained checkpoint
- `build_subgraph_cache.py`: precompute subgraph cache
- `configs.py`: model and training hyperparameters
- `dataloader.py`: data loading, graph building, cache logic
- `models.py`, `gcn.py`, `ban.py`: model definition
- `trainer.py`: training, validation, testing, result saving
- `datasets/`: dataset directory
- `result/`: output directory
- `requirements.txt`: main project dependencies

## 2. Environment

Recommended environment: existing Conda env `SGBan`.

Activate it:

```powershell
conda activate SGBan
```

Install dependencies if needed:

```powershell
pip install -r requirements.txt
```

Notes:

- `torch`, `dgl`, and `torch-scatter` include CUDA-specific versions.
- If these packages are already installed in `SGBan`, you usually do not need to reinstall them.

## 3. Data Layout

The code reads data from:

```text
datasets/<data>/<split>/
```

Example:

```text
datasets/
  biosnap/
    random/
      train.csv
      val.csv
      test.csv
    unseen_drug/
      train.csv
      val.csv
      test.csv
  bindingdb/
    random/
      train.csv
      val.csv
      test.csv
```

Each CSV should contain at least these columns:

- `SMILES`
- `Protein`
- `Y`

Meaning:

- `SMILES`: drug SMILES string
- `Protein`: protein sequence
- `Y`: binary label, usually `0` or `1`

## 4. Recommended Workflow

Run the project in this order.

### 4.1 Build subgraph cache

Before the first training run, precompute cache:

```powershell
python build_subgraph_cache.py --data biosnap --split random --hop 2 --num-workers 1
```

Common arguments:

- `--data`: `bindingdb`, `biosnap`, `human`
- `--split`: `random`, `cold`, `cluster`, `unseen_drug`
- `--hop`: k-hop size, default `2`
- `--num-workers`: number of workers
- `--worker-type`: `process` or `thread`
- `--force`: rebuild cache even if it exists

Cache path:

```text
datasets/subgraph_cache/<data>/<split>/hop_<hop>/
```

### 4.2 Train

Example:

```powershell
python main.py --data biosnap --split random --hop 2
```

The training script automatically runs these seeds:

```text
42, 52, 62, 72, 82
```

For each seed it will:

- load `train.csv`, `val.csv`, `test.csv`
- train the model
- select the best epoch by validation AUROC
- report final test metrics

### 4.3 Test a saved checkpoint

Test the best model of one seed:

```powershell
python test.py --data biosnap --split random --hop 2 --seed 42
```

Or specify a checkpoint manually:

```powershell
python test.py --checkpoint result/biosnap_random_hop2/seed_42/best_model_epoch_XX.pth
```

## 5. Output Files

Training output is saved to:

```text
result/<data>_<split>_hop<hop>/seed_<seed>/
```

Typical files in each seed directory:

- `best_model_epoch_*.pth`
- `model_epoch_*.pth`
- `result_metrics.pt`
- `model_architecture.txt`
- `train_markdowntable.txt`
- `valid_markdowntable.txt`
- `test_markdowntable.txt`
- `plots/threshold_metrics.png`
- `plots/roc_curve.png`
- `plots/pr_curve.png`

After all 5 seeds finish, summary files are saved to:

```text
result/<data>_<split>_hop<hop>/
```

Summary files:

- `seed_summary.csv`
- `seed_summary_stats.csv`

## 6. Example Commands

### BioSNAP random split

```powershell
python build_subgraph_cache.py --data biosnap --split random --hop 2
python main.py --data biosnap --split random --hop 2
python test.py --data biosnap --split random --hop 2 --seed 42
```

### BioSNAP unseen_drug split

```powershell
python build_subgraph_cache.py --data biosnap --split unseen_drug --hop 2
python main.py --data biosnap --split unseen_drug --hop 2
```

### BindingDB random split

```powershell
python build_subgraph_cache.py --data bindingdb --split random --hop 2
python main.py --data bindingdb --split random --hop 2
```

## 7. Key Hyperparameters

Main hyperparameters are in `configs.py`:

- `DRUG.NODE_IN_FEATS`
- `DRUG.HIDDEN_LAYERS`
- `DRUG.MAX_NODES`
- `PROTEIN.NUM_FILTERS`
- `PROTEIN.KERNEL_SIZE`
- `DECODER.HIDDEN_DIM`
- `SOLVER.MAX_EPOCH`
- `SOLVER.BATCH_SIZE`
- `SOLVER.LR`
- `SOLVER.SEED`

Edit `configs.py` and rerun training to apply changes.

## 8. Known Issues In Current Repo

These points come from the current code and directory layout:

- Argument choices include `human`, `cold`, and `cluster`, but the corresponding dataset folders are not currently visible under `datasets/`. Running them as-is may fail because CSV files are missing.
- `requirements.txt` only keeps the main project packages, not the full Conda runtime stack.

## 9. Minimal Repro Run

If you want the shortest runnable path:

```powershell
conda activate SGBan
python build_subgraph_cache.py --data biosnap --split random --hop 2
python main.py --data biosnap --split random --hop 2
```

Then check:

```text
result/biosnap_random_hop2/
```

to see per-seed models and summary results.

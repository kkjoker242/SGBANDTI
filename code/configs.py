from yacs.config import CfgNode as CN

_C = CN()

# Drug feature extractor
_C.DRUG = CN()
_C.DRUG.NODE_IN_FEATS = 75
_C.DRUG.PADDING = True
_C.DRUG.HIDDEN_LAYERS = 128
_C.DRUG.NODE_IN_EMBEDDING = 128
_C.DRUG.MAX_NODES = 290
_C.DRUG.NUM_LAYERS = 3

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]
_C.PROTEIN.KERNEL_SIZE = [3, 6, 9]
_C.PROTEIN.EMBEDDING_DIM = 128
_C.PROTEIN.PADDING = True

# BCN setting
_C.BCN = CN()
_C.BCN.HEADS = 2

# Ablation settings (项9: 2×2 factor design)
_C.ABLATION = CN()
_C.ABLATION.USE_SUBGRAPH = True   # False -> standard GCN drug encoder (整分子池化)
_C.ABLATION.USE_BAN = True        # False -> concat+MLP fusion

# MLP decoder
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 128
_C.DECODER.BINARY = 1
_C.DECODER.DROPOUT = 0.0

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 150
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.LR = 5e-5
_C.SOLVER.SEED = 42
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.COSINE = False

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result"
_C.RESULT.SAVE_MODEL = True

def get_cfg_defaults():
    return _C.clone()


def get_sweep_space():
    return {key: value[:] for key, value in _SWEEP_SPACE.items()}

import os
import torch

# Paths
ROOT_DIR=os.path.dirname(__file__)
SAVE_PATH_BASELINE=os.path.join(ROOT_DIR, 'weights/baseline')
SAVE_PATH_GRU=os.path.join(ROOT_DIR, 'weights/gru')
SAVE_PATH_TRANSFORMER=os.path.join(ROOT_DIR, 'weights/transformer')
TEST_DIR=os.path.join(ROOT_DIR, "test")

# Cross Validation
N_SPLITS=5
RANDOM_STATE=0
FOLD_N=0 # Change the fold number in NN training

# WD coefficient of GRU
WEIGHT_DECAY=0

# BATCH_SIZES
BATCH_SIZE_GRU_SUBJ=4096
BATCH_SIZE_GRU_POL=512
BATCH_SIZE_TRANSFORMER_SUBJ=64
BATCH_SIZE_TRANSFORMER_POL=16

# EPOCHS
EPOCHS_GRU=50
EPOCHS_TRANSFORMER=10

# LEARNING RATES
LR_GRU=10e-3
LR_TRANSFORMER=5e-5

# GRU SETTINGS
EMBEDDING_SIZE=256
GRU_SIZE=128
PAD_TOKEN=0
ATTENTION=True

# MISC
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
SAVE=True if DEVICE == "cuda" else False
FILTER=False # Whether to filter or not objective sentences
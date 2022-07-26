import os
import torch

ROOT_DIR=os.path.dirname(__file__)
SAVE_PATH_BASELINE=os.path.join(ROOT_DIR, 'weights/baseline')
SAVE_PATH_GRU=os.path.join(ROOT_DIR, 'weights/gru')
WEIGHT_DECAY=0
BATCH_SIZE_GRU_SUBJ=4096
BATCH_SIZE_GRU_POL=512
EPOCHS=50
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
LR=10e-3
LR_TRANSFORMER=10e-5
EMBEDDING_SIZE=256
GRU_SIZE=128
PAD_TOKEN=0
ATTENTION=False
FILTER=True
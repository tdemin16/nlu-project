import torch

GENERATOR_SEED=0
SAVE_PATH_BASELINE='weights/baseline'
SAVE_PATH_GRU='weights/gru'
WEIGHT_DECAY=0
BATCH_SIZE=4096
EPOCHS=100
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
LR=10e-3
LR_TRANSFORMER=10e-5
EMBEDDING_SIZE=256
GRU_SIZE=128
PAD_TOKEN=0
ATTENTION=False
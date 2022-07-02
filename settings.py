import torch

GENERATOR_SEED = 0
SAVE_PATH_BASELINE = 'weights/baseline'
WEIGHT_DECAY = 10e-1
BATCH_SIZE = 128
EPOCHS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR=10e-2
MODEL_SIZE=256
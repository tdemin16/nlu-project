import torch

GENERATOR_SEED = 0
SAVE_PATH_BASELINE = 'weights/baseline'
WEIGHT_DECAY = 10e-1
BATCH_SIZE = 32
EPOCHS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 10e-3
LR_TRANSFORMER = 10e-5
EMBEDDING_SIZE = 256
GRU_SIZE = 128
PAD_TOKEN = 0
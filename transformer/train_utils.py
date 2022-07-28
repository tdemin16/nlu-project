import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch import nn
from tqdm.auto import tqdm

from settings import DEVICE
from utils import metrics

def train(model, train_dl, optimizer):
    cum_loss = 0.
    cum_acc = 0.
    cum_f1 = 0.
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for x, a, y in tqdm(train_dl, leave=False):
        optimizer.zero_grad()

        x = x.to(DEVICE)
        a = a.to(DEVICE)
        y = y.type(torch.float).to(DEVICE)

        y_est = model(x, attention_mask=a).logits

        loss = loss_fn(y_est, y.unsqueeze(-1))
        loss.backward()
        optimizer.step()

        cum_loss += loss.item()
        acc, f1 = metrics(torch.sigmoid(y_est), y)
        cum_acc += acc
        cum_f1 += f1

    return cum_loss / len(train_dl), cum_acc / len(train_dl), cum_f1 / len(train_dl)


@torch.no_grad()
def evaluate(model, val_dl):
    cum_loss = 0.
    cum_acc = 0.
    cum_f1 = 0.
    loss_fn = nn.BCEWithLogitsLoss()

    model.eval()
    for x, a, y in val_dl:
        x = x.to(DEVICE)
        a = a.to(DEVICE)
        y = y.type(torch.float).to(DEVICE)
        
        y_est = model(x, attention_mask=a).logits

        loss = loss_fn(y_est, y.unsqueeze(-1))

        cum_loss += loss.item()
        acc, f1 = metrics(torch.sigmoid(y_est), y)
        cum_acc += acc
        cum_f1 += f1

    return cum_loss / len(val_dl), cum_acc / len(val_dl), cum_f1 / len(val_dl)
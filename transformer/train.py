import os
import sys
import time
import torch
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.corpus import subjectivity
from torch import optim, nn
from torch.utils.data import DataLoader

from dataset import SubjectivityDataset
from model import Transformer
from settings import BATCH_SIZE, DEVICE, EPOCHS, LR, LR_TRANSFORMER, WEIGHT_DECAY
from utils import split_dataset, acc


def train(model, train_dl, optimizer):
    cum_loss = 0
    cum_acc = 0

    model.train()
    for x, a, y in tqdm(train_dl, leave=False):
        optimizer.zero_grad()

        x = x.to(DEVICE)
        a = a.to(DEVICE)
        y = y.type(torch.float).to(DEVICE)

        y_est = model(x, a)

        loss = nn.BCELoss()(y_est, y.unsqueeze(-1))
        
        loss.backward()
        optimizer.step()

        cum_loss += loss.item()
        cum_acc += acc(y_est, y)

    return cum_loss / len(train_dl), cum_acc / len(train_dl)


@torch.no_grad()
def evaluate(model, val_dl):
    cum_loss = 0
    cum_acc = 0

    model.eval()
    for x, a, y in val_dl:
        x = x.to(DEVICE)
        a = a.to(DEVICE)
        y = y.type(torch.float).to(DEVICE)
        
        y_est = model(x, a)

        loss = nn.BCELoss()(y_est, y.unsqueeze(-1))

        cum_loss += loss.item()
        cum_acc += acc(y_est, y)

    return cum_loss / len(val_dl), cum_acc / len(val_dl)


def main():
    print("### Transformer subjectivity training ###")
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')
    labels = [0] * len(obj) + [1] * len(subj)
    print(f"\tDataset Size: {len(labels)}")

    train_set, y_train, val_set, y_val, test_set, y_test = split_dataset(obj + subj, labels)
    print(f"\tTrain: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    print(f"### Build Datasets ###")
    train_set = SubjectivityDataset(train_set, y_train)
    val_set = SubjectivityDataset(val_set, y_val)
    test_set = SubjectivityDataset(test_set, y_test)

    print("### Build Dataloaders ###")
    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE)

    model = Transformer().to(DEVICE)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=LR_TRANSFORMER, weight_decay=WEIGHT_DECAY)

    print("### Start Training ###")
    for i in range(EPOCHS):
        start = time.time()

        loss_tr, acc_tr = train(model, train_dl, optimizer)
        
        loss_val, acc_val = evaluate(model, val_dl)

        print(f"""Epoch {i+1}
\tTrain Loss: {loss_tr:.3f}\tTrain Acc: {acc_tr:.3f}
\tValidation Loss: {loss_val:.3f}\tValidation Acc: {acc_val:.3f}
\tElapsed: {time.time() - start:.3f}\n""")


if __name__ == "__main__":
    main()
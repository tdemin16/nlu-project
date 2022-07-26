import copy
import os
import sys
import time
import torch
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.corpus import subjectivity
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import SubjectivityDataset
from model import Transformer
from settings import BATCH_SIZE_TRANSFORMER_SUBJ, DEVICE, EPOCHS, LR_TRANSFORMER, SAVE_PATH_TRANSFORMER
from utils import split_dataset, acc, make


def train(model, train_dl, optimizer):
    cum_loss = 0
    cum_acc = 0
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for x, a, y in tqdm(train_dl, leave=False):
        optimizer.zero_grad()

        x = x.to(DEVICE)
        a = a.to(DEVICE)
        y = y.type(torch.float).to(DEVICE)

        y_est = model(x, a)

        loss = loss_fn(y_est, y.unsqueeze(-1))
        loss.backward()
        optimizer.step()

        cum_loss += loss.item()
        cum_acc += acc(torch.sigmoid(y_est), y)

    return cum_loss / len(train_dl), cum_acc / len(train_dl)


@torch.no_grad()
def evaluate(model, val_dl):
    cum_loss = 0
    cum_acc = 0
    loss_fn = nn.BCEWithLogitsLoss()

    model.eval()
    for x, a, y in val_dl:
        x = x.to(DEVICE)
        a = a.to(DEVICE)
        y = y.type(torch.float).to(DEVICE)
        
        y_est = model(x, a)

        loss = loss_fn(y_est, y.unsqueeze(-1))

        cum_loss += loss.item()
        cum_acc += acc(torch.sigmoid(y_est), y)

    return cum_loss / len(val_dl), cum_acc / len(val_dl)


def main():
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')
    if DEVICE != "cuda":
        #? Reduces the size of the dataset when running on CPU.
        #? Less comuputational requirements during test
        obj = obj[:20]
        subj = subj[:20]
        print("[Warning] Cuda not detected, a subset of the dataset will be used.")

    labels = [0] * len(obj) + [1] * len(subj)
    train_set, y_train, test_set, y_test = split_dataset(obj + subj, labels)

    train_set = SubjectivityDataset(train_set, y_train)
    test_set = SubjectivityDataset(test_set, y_test)

    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE_TRANSFORMER_SUBJ, shuffle=True)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE_TRANSFORMER_SUBJ)

    model = Transformer().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR_TRANSFORMER)

    best_model = None
    best_test_acc = 0
    for i in range(EPOCHS):
        start = time.time()
        new_best = False

        loss_tr, acc_tr = train(model, train_dl, optimizer)
        loss_test, acc_test = evaluate(model, test_dl)

        if acc_test > best_test_acc:
            best_model = copy.deepcopy(model)
            best_test_acc = acc_test
            new_best = True

        print(f"Epoch {i+1}")
        print(f"\tTrain Loss: {loss_tr:.2f}\tTrain Acc: {acc_tr:.2f}")
        print(f"\tTest Loss: {loss_test:.2f}\tTest Acc: {acc_test:.2f}")
        print(f"\tElapsed: {time.time() - start:.2f}")
        if new_best: print("\tNew Best Model")
        
        print("")

    loss_ts, acc_ts = evaluate(best_model, test_dl)
    print(f"Best weights")
    print(f"Loss: {loss_ts:.2f} - Acc: {acc_ts:.2f}")

    make(SAVE_PATH_TRANSFORMER)
    torch.save(best_model.state_dict(), os.path.join(SAVE_PATH_TRANSFORMER, "subj.pth"))


if __name__ == "__main__":
    main()
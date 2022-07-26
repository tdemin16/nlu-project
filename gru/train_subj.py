import copy
import os
import pickle
import sys
import time
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.corpus import subjectivity
from torch import optim, nn
from torch.utils.data import DataLoader

from dataset import SubjectivityDataset
from model import GRUAttention
from utils import acc, collate_fn, make, make_w2id, split_dataset, init_weights
from settings import SAVE, SAVE_PATH_GRU, WEIGHT_DECAY, BATCH_SIZE_GRU_SUBJ, EPOCHS, DEVICE, LR


def train(model, train_dl, optimizer):
    cum_loss = 0.
    cum_acc = 0.
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for x, y, l in train_dl:
        optimizer.zero_grad()
        
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        l = l.to(DEVICE)

        y_est = model(x, l)

        loss = loss_fn(y_est, y.unsqueeze(-1))
        loss.backward()
        optimizer.step()

        cum_loss += loss.item()
        cum_acc += acc(torch.sigmoid(y_est), y)

    return cum_loss / len(train_dl), cum_acc / len(train_dl)


@torch.no_grad()
def evaluate(model, val_dl):
    cum_loss = 0.
    cum_acc = 0.
    loss_fn = nn.BCEWithLogitsLoss()

    model.eval()
    for x, y, l in val_dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        l = l.to(DEVICE)
        
        y_est = model(x, l)

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
    X_train, y_train, X_test, y_test = split_dataset(obj + subj, labels)

    vocab = set([w for d in X_train for w in d])
    w2id, w2id_size = make_w2id(vocab)

    train_set = SubjectivityDataset(X_train, y_train, w2id)
    test_set = SubjectivityDataset(X_test, y_test, w2id)
    
    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE_GRU_SUBJ, collate_fn=collate_fn, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE_GRU_SUBJ   , collate_fn=collate_fn)

    model = GRUAttention(num_embeddings=w2id_size).to(DEVICE)
    model.apply(init_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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
        print(f"\tTrain Loss: {loss_tr:.3f}\tTrain Acc: {acc_tr:.3f}")
        print(f"\tTest Loss: {loss_test:.3f}\tTest Acc: {acc_test:.3f}")
        print(f"\tElapsed: {time.time() - start:.3f}")
        if new_best: print("\tNew Best Model")
        
        print("")

    loss_ts, acc_ts = evaluate(best_model, test_dl)
    print(f"Best weights")
    print(f"Loss: {loss_ts:.3f} - Acc: {acc_ts:.3f}")

    if SAVE:
        make(SAVE_PATH_GRU)
        torch.save(best_model.state_dict(), os.path.join(SAVE_PATH_GRU, "subj_cls.pth"))
        with open(os.path.join(SAVE_PATH_GRU, "subj_w2id.pkl"), 'wb') as f:
            pickle.dump(w2id, f)
        print("Weights saved")

if __name__ == "__main__":
    main()
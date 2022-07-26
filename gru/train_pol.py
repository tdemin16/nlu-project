import copy
import os
import pickle
import sys
import time
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.corpus import movie_reviews
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import PolarityDataset
from model import GRUAttention
from settings import BATCH_SIZE_GRU_POL, DEVICE, EPOCHS, FILTER, LR, SAVE_PATH_GRU, WEIGHT_DECAY
from utils import acc, collate_fn, init_weights, make, make_w2id, remove_objective_sents_nn, split_dataset


def train(model, optimizer, train_dl):
    cum_loss = 0.
    cum_acc = 0.
    loss_fn = torch.nn.BCEWithLogitsLoss()

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
    loss_fn = torch.nn.BCEWithLogitsLoss()

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
    neg = movie_reviews.paras(categories='neg')
    pos = movie_reviews.paras(categories='pos')
    if DEVICE != "cuda":
        # ? Reduces the size of the dataset when running on CPU.
        # ? Less comuputational requirements during test
        neg = neg[:20]
        pos = pos[:20]
        print("[Warning] Cuda not detected, a subset of the dataset will be used.")
    
    targets = [0] * len(neg) + [1] * len(pos)

    if FILTER:
        with open(os.path.join(SAVE_PATH_GRU, 'subj_w2id.pkl'), 'rb') as f:
            subj_w2id = pickle.load(f)
        subj_det = GRUAttention(num_embeddings=len(subj_w2id.keys()))
        subj_det.load_state_dict(torch.load(
            os.path.join(SAVE_PATH_GRU, 'subj_cls.pth'), map_location=DEVICE)
        )
        subj_det.to(DEVICE)
        subj_det.eval()

        filt_neg = []
        for doc in neg:
            filt_doc = remove_objective_sents_nn(subj_det, subj_w2id, doc)
            if len(filt_doc) > 0:
                filt_neg.append(filt_doc)

        filt_pos = []
        for doc in pos:
            filt_doc = remove_objective_sents_nn(subj_det, subj_w2id, doc)
            if len(filt_doc) > 0:
                filt_pos.append(filt_doc)

        neg = filt_neg
        pos = filt_pos
        targets = [0] * len(filt_neg) + [1] * len(filt_pos)

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(neg + pos, targets)
    
    vocab = set([w for doc in X_train for sent in doc for w in sent])
    w2id, w2id_size = make_w2id(vocab)

    train_set = PolarityDataset(X_train, y_train, w2id)
    val_set = PolarityDataset(X_val, y_val, w2id)
    test_set = PolarityDataset(X_test, y_test, w2id)

    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE_GRU_POL, collate_fn=collate_fn, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_set, batch_size=BATCH_SIZE_GRU_POL, collate_fn=collate_fn)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE_GRU_POL, collate_fn=collate_fn)

    model = GRUAttention(num_embeddings=w2id_size).to(DEVICE)
    model.apply(init_weights)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_model = None
    best_val_acc = 0
    for i in range(EPOCHS):
        start = time.time()
        new_best = False

        loss_tr, acc_tr = train(model, optimizer, train_dl)
        loss_val, acc_val = evaluate(model, val_dl)

        if acc_val > best_val_acc:
            best_model = copy.deepcopy(model)
            best_val_acc = acc_val
            new_best = True

        print(f"Epoch {i+1}")
        print(f"\tTrain Loss: {loss_tr:.2f}\tTrain Acc: {acc_tr:.2f}")
        print(f"\tValidation Loss: {loss_val:.2f}\tValidation Acc: {acc_val:.2f}")
        print(f"\tElapsed: {time.time() - start:.2f}")
        if new_best: print("\tNew Best Model")
        
        print("")

    loss_ts, acc_ts = evaluate(best_model, test_dl)
    print(f"Performances on the Test Set")
    print(f"Loss: {loss_ts:.2f} - Acc: {acc_ts:.2f}")

    make(SAVE_PATH_GRU)
    torch.save(best_model.state_dict(), os.path.join(SAVE_PATH_GRU, "pol.pth"))

if __name__ == "__main__":
    main()
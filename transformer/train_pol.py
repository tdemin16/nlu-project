import copy
import os
import sys
import time
import torch
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.corpus import movie_reviews
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from dataset import PolarityDataset
from settings import BATCH_SIZE_TRANSFORMER_SUBJ, DEVICE, EPOCHS_TRANSFORMER, FILTER, LR_TRANSFORMER, SAVE, SAVE_PATH_TRANSFORMER
from utils import remove_objective_sents_transformer, split_dataset, acc, make


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
    neg = movie_reviews.paras(categories='neg')
    pos = movie_reviews.paras(categories='pos')
    if DEVICE != "cuda":
        #? Reduces the size of the dataset when running on CPU.
        #? Less comuputational requirements during test
        neg = neg[:10]
        pos = pos[:10]
        print("[Warning] Cuda not detected, a subset of the dataset will be used.")

    if FILTER:
        subj_det = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(SAVE_PATH_TRANSFORMER, "subj.pth"),
            ignore_mismatched_size=True
        ).to(DEVICE)
        subj_det.eval()

        filt_neg = []
        for doc in neg:
            filt_doc = remove_objective_sents_transformer(subj_det, doc)
            if len(filt_doc) > 0:
                filt_neg.append(filt_doc)

        filt_pos = []
        for doc in pos:
            filt_doc = remove_objective_sents_transformer(subj_det, doc)
            if len(filt_doc) > 0:
                filt_pos.append(filt_doc)

        neg = filt_neg
        pos = filt_pos
    
    targets = [0] * len(neg) + [1] * len(pos)
    train_set, y_train, test_set, y_test = split_dataset(neg + pos, targets)

    train_set = PolarityDataset(train_set, y_train)
    test_set = PolarityDataset(test_set, y_test)

    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE_TRANSFORMER_SUBJ, shuffle=True)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE_TRANSFORMER_SUBJ)

    model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            num_labels=1,
            ignore_mismatched_sizes=True
        ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR_TRANSFORMER)

    best_model = None
    best_test_acc = 0
    for i in range(EPOCHS_TRANSFORMER):
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
        make(SAVE_PATH_TRANSFORMER)
        best_model.save_pretrained(os.path.join(SAVE_PATH_TRANSFORMER, "pol.pth"))
        print("Weights saved")


if __name__ == "__main__":
    main()
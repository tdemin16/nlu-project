import os
import sys
import time
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.corpus import movie_reviews
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import PolarityDataset
from model import GRUAttention
from settings import BATCH_SIZE, DEVICE, EPOCHS, LR, PAD_TOKEN, WEIGHT_DECAY
from utils import acc, split_dataset


def train(model, optimizer, train_dl):
    cum_loss = 0.
    cum_acc = 0.
    loss_fn = torch.nn.BCELoss()

    model.train()
    for x, y, l in train_dl:
        optimizer.zero_grad()

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        l = l.to(DEVICE)

        y_hat = model(x, l)

        loss = loss_fn(y_hat, y.unsqueeze(-1))
        loss.backward()
        optimizer.step()

        cum_loss += loss
        cum_acc += acc(y_hat, y)

    return cum_loss / len(train_dl), cum_acc / len(train_dl)


@torch.no_grad()
def evaluate(model, val_dl):
    cum_loss = 0.
    cum_acc = 0.
    loss_fn = torch.nn.BCELoss()

    model.eval()
    for x, y, l in val_dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        l = l.to(DEVICE)

        y_hat = model(x, l)

        loss = loss_fn(y_hat, y.unsqueeze(-1))

        cum_loss += loss
        cum_acc += acc(y_hat, y)

    return cum_loss / len(val_dl), cum_acc / len(val_dl)


def main():
    neg = movie_reviews.paras(categories='neg')
    pos = movie_reviews.paras(categories='pos')
    if DEVICE != "cuda":
        # ? Dataset very big, it avoids to run everything on my laptop
        neg = neg[:100]
        pos = pos[:100]
    targets = [0] * len(neg) + [1] * len(pos)

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(neg + pos, targets)

    w2id, size_w2id = make_vocab(X_train)

    train_set = PolarityDataset(X_train, y_train, w2id)
    val_set = PolarityDataset(X_val, y_val, w2id)
    test_set = PolarityDataset(X_test, y_test, w2id)

    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = GRUAttention(num_embeddings=size_w2id).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for i in range(EPOCHS):
        start = time.time()

        loss_tr, acc_tr = train(model, optimizer, train_dl)
        loss_val, acc_val = evaluate(model, val_dl)

        print(f"""Epoch {i+1}
\tTrain Loss: {loss_tr:.3f}\tTrain Acc: {acc_tr:.3f}
\tValidation Loss: {loss_val:.3f}\tValidation Acc: {acc_val:.3f}
\tElapsed: {time.time() - start:.3f}\n""")

if __name__ == "__main__":
    main()
import os
import sys
import time
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import SubjectivityDataset

from nltk.corpus import subjectivity
from torch import optim, nn
from torch.utils.data import DataLoader

from model import GRUAttention
from utils import acc, collate_fn, make_vocab, split_dataset, init_weights
from settings import WEIGHT_DECAY, BATCH_SIZE, EPOCHS, DEVICE, LR


def train(model, train_dl, optimizer):
    cum_loss = 0
    cum_acc = 0

    model.train()
    for x, y, l in train_dl:
        optimizer.zero_grad()

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        l = l.to(DEVICE)

        y_est = model(x, l)

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
    for x, y, l in val_dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        l = l.to(DEVICE)
        
        y_est = model(x, l)

        loss = nn.BCELoss()(y_est, y.unsqueeze(-1))

        cum_loss += loss.item()
        cum_acc += acc(y_est, y)

    return cum_loss / len(val_dl), cum_acc / len(val_dl)


def main():
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')
    labels = [0] * len(obj) + [1] * len(subj)
    train_set, y_train, val_set, y_val, test_set, y_test = split_dataset(obj + subj, labels)

    vocab, vocab_size = make_vocab(train_set)

    train_set = SubjectivityDataset(train_set, y_train, vocab)
    val_set = SubjectivityDataset(val_set, y_val, vocab)
    test_set = SubjectivityDataset(test_set, y_test, vocab)
    
    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = GRUAttention(num_embeddings=vocab_size).to(DEVICE)
    model.apply(init_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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
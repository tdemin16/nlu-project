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


def collate_fn(batch):
    # adapted from labs
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # Matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort batch by seq length
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    X = [x for x, y in batch]
    y = [y for x, y in batch]
    
    X, lengths = merge(X)
    y = torch.FloatTensor(y)
    lengths = torch.LongTensor(lengths)

    return X, y, lengths


def make_vocab(data):
    vocab = set([w for doc in data for sent in doc for w in sent])
    w2id = {"pad": PAD_TOKEN}
    for w in vocab:
        w2id[w] = len(w2id)

    w2id["unk"] = len(w2id)
    return w2id, len(w2id)


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
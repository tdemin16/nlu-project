import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import SubjectivityDataset

from nltk.corpus import subjectivity
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy

from model import GRUAttention
from utils import acc, collate_fn, get_text_pipline, split_dataset, init_weights
from settings import WEIGHT_DECAY, BATCH_SIZE, EPOCHS, DEVICE, LR


def train(model, train_dl, optimizer):
    cum_loss = 0
    cum_acc = 0
    num_samples = 0

    model.train()
    for x, y in train_dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y_est = model(x)

        loss = binary_cross_entropy(y_est, y.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        cum_loss += loss.item()
        cum_acc += acc(y_est, y)
        num_samples += x.size(0)

    return cum_loss / num_samples, cum_acc / num_samples


@torch.no_grad()
def evaluate(model, val_dl):
    cum_loss = 0
    cum_acc = 0
    num_samples = 0

    model.eval()
    for x, y in val_dl:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y_est = model(x)

        loss = binary_cross_entropy(y_est, y.unsqueeze(-1))

        cum_loss += loss.item()
        cum_acc += acc(y_est, y)
        num_samples += x.size(0)

    return cum_loss / num_samples, cum_acc / num_samples


def main():
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')
    labels = [0] * len(obj) + [1] * len(subj)
    train_set, y_train, val_set, y_val, test_set, y_test = split_dataset(obj + subj, labels)

    text_pipeline, vocab_size = get_text_pipline(train_set)

    train_set = SubjectivityDataset(train_set, y_train, text_pipeline)
    val_set = SubjectivityDataset(val_set, y_val, text_pipeline)
    test_set = SubjectivityDataset(test_set, y_test, text_pipeline)
    
    train_dl = DataLoader(train_set, BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = GRUAttention(num_embeddings=vocab_size)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for i in range(EPOCHS):
        loss_tr, acc_tr = train(model, train_dl, optimizer)
        loss_val, acc_val = evaluate(model, val_dl)
        print(f"Epoch {i+1}\n\tTrain Loss: {loss_tr:.3f}\tTrain Acc: {acc_tr:.3f}\n\tValidation Loss: {loss_val:.3f}\tValidation Acc: {acc_val:.3f}")


if __name__ == "__main__":
    main()
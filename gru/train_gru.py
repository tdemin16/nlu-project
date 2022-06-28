import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import SubjectivityDataset

from nltk.corpus import subjectivity
from torch import nn, optim
from torch.utils.data import DataLoader

from model import GRUAttention
from utils import split_dataset, init_weights, make_vocab
from settings import WEIGHT_DECAY, BATCH_SIZE, EPOCHS, DEVICE


def train(model, train_dl, optimizer, loss):
    cum_loss = 0
    cum_acc = 0
    num_samples = 0

    model.train()
    for x, y in train_dl:

        y = y.to(DEVICE)

        y_est = model(x)

        optimizer.zero_grad()


def main():
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')
    labels = [-1] * len(obj) + [1] * len(subj)
    train_set, y_train, val_set, y_val, test_set, y_test = split_dataset(obj + subj, labels)
    
    # convert string to integers
    # adapted from https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    vocab = make_vocab(set([w for d in train_set for w in d]))
    text_pipeline = lambda x: [vocab[w] if w in vocab.keys() else vocab['<unk>'] for w in x]

    train_set = SubjectivityDataset(train_set, y_train, text_pipeline)
    val_set = SubjectivityDataset(val_set, y_val, text_pipeline)
    test_set = SubjectivityDataset(test_set, y_test, text_pipeline)
    
    train_dl = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE)

    model = GRUAttention(num_embeddings=len(train_set))
    model.apply(init_weights)

    loss = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY)

    exit()
    for i in range(EPOCHS):
        train(model, train_dl, optimizer, loss)


if __name__ == "__main__":
    main()
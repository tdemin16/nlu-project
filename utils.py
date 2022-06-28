import os
import torch

from sklearn.model_selection import train_test_split
from torch import nn

from settings import GENERATOR_SEED


def collate_batch(batch):
    """
    Adapted from torch guide: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    """
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(_label)


def make_vocab(data):
    """
    Return a mapping word to integer.
    """
    assert type(data) == set
    vocab = {w: i for i, w in enumerate(data)}
    vocab['<unk>'] = len(data)
    return vocab


def list2str(l):
    """
    Returns a string from a list
    """
    return ' '.join(w for w in l)


def remove_objective_sents(classifier, vectorizer, document):
    """
    Remove the objective sentences from a document and returns the filtered document.
    """
    document = [list2str(p) for p in document]
    vectors = vectorizer.transform(document)
    estimated_subj = classifier.predict(vectors)
    filt_sent = [d for d, est in zip(document, estimated_subj) if est == 1]
    filt_doc = list2str(filt_sent)
    return filt_doc


def make(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_dataset(dataset, labels):
    """
    Split dataset in train, validation and test set.
    Uses GENERATOR_SEED defined in settings to keep the test set fixed across runs.
    """

    inter_set, test_set, inter_labels, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=GENERATOR_SEED)
    train_set, val_set, y_train, y_val = train_test_split(inter_set, inter_labels, test_size=0.2/0.8)
    
    return train_set, y_train, val_set, y_val, test_set, y_test


def init_weights(mat):
    """
    Weight initialization as explained during labs.
    """
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)
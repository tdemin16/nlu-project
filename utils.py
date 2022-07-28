import os
import numpy as np
import torch

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader

from lstm import dataset as lstm_ds
from settings import DEVICE, FOLD_N, N_SPLITS, PAD_TOKEN, RANDOM_STATE
from transformer import dataset as trans_ds


class FilteredData:
    def __init__(self, neg, pos):
        self.neg = neg
        self.pos = pos


def make_w2id(vocab):
    """
    Return a mapping word to integer.
    Adapted from labs.
    """
    w2id = {"pad": PAD_TOKEN}
    for w in vocab:
        w2id[w] = len(w2id)

    w2id["unk"] = len(w2id)
    return w2id, len(w2id.keys())


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


def acc(y_est, y):
    """
    Compute the accuracy
    """
    return accuracy_score(y.cpu().detach().numpy(), torch.round(y_est).cpu().detach().numpy())


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


@torch.no_grad()
def remove_objective_sents_nn(classifier, w2id, document):
    """
    Remove the objective sentences from a document and returns the filtered document.
    """
    ds = lstm_ds.SubjectivityDataset(document, [0]*len(document), w2id)
    dl = DataLoader(ds, batch_size=len(document), shuffle=False, collate_fn=collate_fn)
    it = dl.__iter__()
    x, y, l = it.__next__()

    x = x.to(DEVICE)
    l = l.to(DEVICE)

    est_subj = torch.sigmoid(classifier(x, l))
    est_subj = torch.round(est_subj).cpu().detach()
    filt_doc = [s for s, est in zip(document, est_subj) if est == 1]
    return filt_doc


@torch.no_grad()
def remove_objective_sents_transformer(classifier, document):
    """
    Remove the objective sentences from a document and returns the filtered document.
    """
    ds = trans_ds.SubjectivityDataset(document, [0]*len(document))
    dl = DataLoader(ds, batch_size=len(document), shuffle=False)
    it = dl.__iter__()
    x, a, y = it.__next__()

    x = x.to(DEVICE)
    a = a.to(DEVICE)

    est_subj = torch.sigmoid(classifier(x, a).logits)
    est_subj = torch.round(est_subj).cpu().detach()
    filt_doc = [s for s, est in zip(document, est_subj) if est == 1]
    return filt_doc


def make(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_dataset(dataset, labels):
    """
    Split dataset in train, validation and test set.
    Uses GENERATOR_SEED defined in settings to keep the test set fixed across runs.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold = list(cv.split(dataset, labels))[FOLD_N]

    dataset = np.array(dataset, dtype=np.object0)
    labels = np.array(labels)

    train_set = dataset[fold[0]]
    y_train = labels[fold[0]]

    test_set = dataset[fold[1]]
    y_test = labels[fold[1]]
    
    return train_set, y_train, test_set, y_test


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
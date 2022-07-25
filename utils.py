from collections import Counter
import os
import torch

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn

from settings import GENERATOR_SEED, PAD_TOKEN


def make_vocab(data):
    """
    Return a mapping word to integer.
    Adapted from labs.
    """
    data = set([w for d in data for w in d])
    vocab = {"pad": PAD_TOKEN}
    count = Counter(data)
    for k in count.keys():
        vocab[k] = len(vocab)

    vocab["unk"] = len(vocab)
    return vocab, len(vocab.keys())


def map_seq(data, vocab):
    """
    Convert string to integers.
    adapted from lab.
    """
    res = []
    for doc in data:
        tmp_doc = []
        for w in doc:
            if w in vocab:
                tmp_doc.append(vocab[w])
            else:
                tmp_doc.append(vocab['unk'])

        res.append(tmp_doc)

    return res


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
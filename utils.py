import os
import torch

from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from settings import GENERATOR_SEED


def make_vocab(data):
    """
    Return a mapping word to integer.
    """
    assert type(data) == set
    # +1 to handle offset == 0
    vocab = {w: i+1 for i, w in enumerate(data)}
    vocab['<unk>'] = len(data) + 1
    return vocab


def get_text_pipline(train_set):
    """
    Convert string to integers.
    adapted from https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    """
    vocab = make_vocab(set([w for d in train_set for w in d]))
    text_pipeline = lambda x: [vocab[w] if w in vocab.keys() else vocab['<unk>'] for w in x]
    return text_pipeline, len(vocab.keys())


def collate_fn(batch):
    # adapted from https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    text_list, label_list = [], []
    for (_text, _label) in batch:
        text_list.append(torch.tensor(_text))
        label_list.append(_label)

    x = pad_sequence(text_list, batch_first=True)
    y = torch.tensor(label_list, dtype=torch.float)
    return x, y


def acc(y_est, y):
    """
    Compute the accuracy
    """
    y_est = torch.round(y_est)
    return (y_est == y).sum()  


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
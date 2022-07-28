import torch

from torch.utils.data import Dataset

class SubjectivityDataset(Dataset):
    """
    0 are objective sentences
    1 are subjective ones
    """
    def __init__(self, data, targets, w2id):
        super().__init__()
        assert len(data) == len(targets), "data and targets length is not the same"
        self.corpus = None
        self.labels = targets

        self._map(data, w2id)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.Tensor(self.corpus[index]), self.labels[index]

    def _map(self, data, w2id):
        self.corpus = []
        for doc in data:
            tmp_doc = []
            for w in doc:
                id = w2id[w] if w in w2id.keys() else w2id["unk"]
                tmp_doc.append(id)

            self.corpus.append(tmp_doc)


class PolarityDataset(Dataset):
    """
    0 are negative documents
    1 are positive once
    """
    def __init__(self, X, y, w2id):
        super().__init__()
        assert len(X) == len(y)
        self.corpus = None
        self.targets = y
        
        self._map(X, w2id)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.tensor(self.corpus[index]), self.targets[index]

    def _map(self, X, w2id):
        self.corpus = []
        for doc in X:
            tmp_doc = []
            for sent in doc:
                for w in sent:
                    id = w2id[w] if w in w2id.keys() else w2id["unk"]
                    tmp_doc.append(id)

            self.corpus.append(tmp_doc)
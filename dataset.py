from torch.utils.data import Dataset

class PolarityDataset(Dataset):
    """
    -1 are negative documents
    1 are positive once
    """
    def __init__(self, data):
        super().__init__()
        neg = data.paras(categories='neg')
        pos = data.paras(categories='pos')
        
        self.corpus = neg + pos
        self.labels = [-1] * len(neg) + [1] * len(pos)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.corpus[index], self.labels[index]


class SubjectivityDataset(Dataset):
    """
    -1 are objective sentences
    1 are subjective ones
    """
    def __init__(self, data):
        super().__init__()
        obj = data.sents(categories='obj')
        subj = data.sents(categories='subj')

        self.corpus = obj + subj
        self.labels = [-1] * len(obj) + [1] * len(subj)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.corpus[index], self.labels[index]

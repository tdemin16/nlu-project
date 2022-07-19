import torch

from torch.utils.data import Dataset
from transformers import BertTokenizer

class SubjectivityDataset(Dataset):
    """
    0 are objective sentences
    1 are subjective ones
    """
    def __init__(self, data, targets, text_pipeline=None):
        super().__init__()
        assert len(data) == len(targets), "data and targets length is not the same"
        self.pipeline = text_pipeline is not None
        if self.pipeline:
            self.corpus = [text_pipeline(d) for d in data]
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            encoding = tokenizer([' '.join(d) for d in data], return_tensors="pt", padding=True)
            self.corpus = encoding['input_ids']
            self.att_mask = encoding['attention_mask']
        self.labels = targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.pipeline:
            return torch.Tensor(self.corpus[index]), self.labels[index]
        else:
            return self.corpus[index], self.att_mask[index], self.labels[index]


class PolarityDataset(Dataset):
    """
    0 are negative documents
    1 are positive once
    """
    def __init__(self, data):
        super().__init__()
        neg = data.paras(categories='neg')
        pos = data.paras(categories='pos')
        
        self.corpus = neg + pos
        self.labels = [0] * len(neg) + [1] * len(pos)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.corpus[index], self.labels[index]
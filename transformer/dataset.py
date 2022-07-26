import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SubjectivityDataset(Dataset):
    """
    0 are objective sentences
    1 are subjective ones
    """
    def __init__(self, data, targets):
        super().__init__()
        assert len(data) == len(targets), "data and targets length is not the same"
        tokenizer = AutoTokenizer.from_pretrained("cffl/bert-base-styleclassification-subjective-neutral")
        encoding = tokenizer([' '.join(d) for d in data], return_tensors="pt", padding=True)
        self.corpus = encoding['input_ids']
        self.att_mask = encoding['attention_mask']
        self.labels = targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.corpus[index], self.att_mask[index], self.labels[index]


class PolarityDataset(Dataset):
    """
    0 are negative documents
    1 are positive once
    """
    # cardiffnlp/twitter-xlm-roberta-base-sentiment
    def __init__(self, data, targets):
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
        encoding = tokenizer(
            [self._lol2str(d) for d in data], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        )
        self.corpus = encoding['input_ids']
        self.att_mask = encoding['attention_mask']
        self.labels = targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.corpus[index], self.att_mask[index], self.labels[index]

    def _list2str(self, l):
        """
        Returns a string from a list
        """
        return ' '.join(w for w in l)

    def _lol2str(self, lol):
        """
        Returns a string from a list of lists
        """
        return ' '.join(self._list2str(l) for l in lol)
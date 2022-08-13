"""
Use this file to run subjectivity detection and polarity classification on custom movie reviews.
"""
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk import sent_tokenize, word_tokenize
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from settings import DEVICE, SAVE_PATH_TRANSFORMER, TEST_DIR
from transformer.dataset import PolarityDataset
from utils import remove_objective_sents_transformer


REVIEWS = os.path.join(TEST_DIR, "reviews.txt") # name of the reviews in the directory TEST_DIR
FILTER = False # whether to filter or not reviews


def main():
    reviews = []
    with open(REVIEWS, 'r') as fp:
        for line in fp:
            # remove '\n'
            sentences = line[:-1].lower()
            # apply tokenization in order to reuse training code
            sentences = [word_tokenize(s) for s in sent_tokenize(sentences)]
            reviews.append(sentences)

    # Load subjectivity detection weights
    subj_det = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(SAVE_PATH_TRANSFORMER, "subj")
    ).to(DEVICE)
    subj_det.eval()

    # Load polarity classification weights
    pol_cls = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(SAVE_PATH_TRANSFORMER, "pol")
    ).to(DEVICE)
    pol_cls.eval()

    # Filter sentences
    if FILTER:
        filt_reviews = []
        for r in reviews:
            filt_reviews.append(remove_objective_sents_transformer(subj_det, r))
            print(filt_reviews[-1], "\n")
        
        reviews = filt_reviews

    # Build Dataset and get a batch
    #! pay attention to the dimension of the batch.
    #! if it is too big it would return and error.
    #! to use bigger batches the code must be changed.
    #! this part is intended to be used with small data.
    ds = PolarityDataset(reviews, [0]*len(reviews))
    dl = DataLoader(ds, batch_size=len(reviews))
    it = dl.__iter__()
    x, a, _ = it.__next__()

    x.to(DEVICE)
    a.to(DEVICE)

    # get estimations
    est_subj = torch.sigmoid(pol_cls(x, a).logits)
    est_subj = torch.round(est_subj).cpu().detach()
    
    # print estimations
    for e in est_subj:
        if e.int().item() == 1:
            print("positive")
        else:
            print("negative")

if __name__ == "__main__":
    main()
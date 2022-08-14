"""
The purpose of this file is to filter out objective sentences using a trained
BERT encoder to spare time duing training of the polarity classifier.
"""
import os
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.corpus import movie_reviews
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm

from settings import DEVICE, ROOT_DIR, SAVE_PATH_TRANSFORMER
from utils import FilteredData, remove_objective_sents_transformer


def main():
    # Get data
    neg = movie_reviews.paras(categories='neg')
    pos = movie_reviews.paras(categories='pos')
    
    #? Reduces the size of the dataset when running on CPU.
    #? Less comuputational requirements during test
    if DEVICE != "cuda":
        neg = neg[:10]
        pos = pos[:10]
        print("[Warning] Cuda not detected, a subset of the dataset will be used.")

    # Load subjectivity detector
    subj_det = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(SAVE_PATH_TRANSFORMER, "subj"),
        ).to(DEVICE)
    subj_det.eval()

    # filter negative sentences
    filt_neg = []
    for doc in tqdm(neg, desc="neg", leave=False):
        filt_doc = remove_objective_sents_transformer(subj_det, doc)
        if len(filt_doc) > 0:
            filt_neg.append(filt_doc)

    # filter positive sentences
    filt_pos = []
    for doc in tqdm(pos, desc="pos", leave=False):
        filt_doc = remove_objective_sents_transformer(subj_det, doc)
        if len(filt_doc) > 0:
            filt_pos.append(filt_doc)

    # Build a unique dataset so it can be stored in a file
    filt_data = FilteredData(filt_neg, filt_pos)
    with open(os.path.join(SAVE_PATH_TRANSFORMER, "filt_data.pkl"), "wb") as fp:
        pickle.dump(filt_data, fp)


if __name__ == "__main__":
    if os.path.isdir(os.path.join(SAVE_PATH_TRANSFORMER, "subj")):
        main()
    else:
        print(f"[ERROR] No subjectivity classification weights found!\nUse {os.path.join(ROOT_DIR, 'transformer', 'train_subj.py')} to train them.")
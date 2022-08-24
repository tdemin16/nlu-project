"""
The purpose of this file is to print out wrongful prediction in the test set in order to analyze them.

Requirements:
 - Filtered dataset
 - Trained transformer in polarity classification
Both placed in the default directory.
"""
import os
import pickle
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from settings import DEVICE, ROOT_DIR, SAVE_PATH_TRANSFORMER
from transformer.dataset import PolarityDataset
from tqdm.auto import tqdm
from utils import lol2str, split_dataset


def main():
    path = os.path.join(SAVE_PATH_TRANSFORMER, "filt_data.pkl")
    if not os.path.exists(path):
        print(f"[ERROR] No filtered data found!\nUse {os.path.join(ROOT_DIR, 'transformer', 'filter_sents.py')} to filter it.")
        exit(1)
    
    with open(path, "rb") as fp:
        filt_data = pickle.load(fp)
    neg = filt_data.neg
    pos = filt_data.pos

    # Compute lebels and split in train/test set
    targets = [0] * len(neg) + [1] * len(pos)
    # fold must be selected in settings.py
    train_set, y_train, test_set, y_test = split_dataset(neg + pos, targets)

    # Load polarity classification weights
    pol_cls = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(SAVE_PATH_TRANSFORMER, "pol")
    ).to(DEVICE)
    pol_cls.eval()

    # Build Dataset and dataloader
    ds = PolarityDataset(test_set, y_test)
    dl = DataLoader(ds, batch_size=1)

    wrong = []
    for i, (x, a, y) in tqdm(enumerate(dl)):
        x.to(DEVICE)
        a.to(DEVICE)
        y = y.type(torch.float).to(DEVICE)

        # get estimations
        y_est = pol_cls(x, a).logits
        y_est = torch.round(torch.sigmoid(y_est)).cpu().detach()

        if y_est != y:
            wrong.append((y_est, y, test_set[i]))

    for y_est, y, doc in wrong:
        print(f"* **Pred: {int(y_est.item())} - GT: {int(y.item())}**: ", lol2str(doc), '\n')

if __name__ == "__main__":
    main()
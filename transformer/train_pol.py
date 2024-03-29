import copy
import os
import pickle
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.corpus import movie_reviews
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from dataset import PolarityDataset
from settings import BATCH_SIZE_TRANSFORMER_POL, DEVICE, EPOCHS_TRANSFORMER, FILTER, FOLD_N, LR_TRANSFORMER, ROOT_DIR, SAVE, SAVE_PATH_TRANSFORMER
from train_utils import evaluate, train
from utils import FilteredData, split_dataset, make


def main():
    # if FILTER == True use filtered dataset
    if FILTER:
        path = os.path.join(SAVE_PATH_TRANSFORMER, "filt_data.pkl")
        if not os.path.exists(path):
            print(f"[ERROR] No filtered data found!\nUse {os.path.join(ROOT_DIR, 'transformer', 'filter_sents.py')} to filter it.")
            exit(1)
        
        with open(path, "rb") as fp:
            filt_data = pickle.load(fp)

        neg = filt_data.neg
        pos = filt_data.pos
    
    else:
        neg = movie_reviews.paras(categories='neg')
        pos = movie_reviews.paras(categories='pos')

    #? Reduces the size of the dataset when running on CPU.
    #? Less comuputational requirements during test
    if DEVICE != "cuda":
        neg = neg[:10]
        pos = pos[:10]
        print("[Warning] Cuda not detected, a subset of the dataset will be used.")
    
    # Compute lebels and split in train/test set
    targets = [0] * len(neg) + [1] * len(pos)
    train_set, y_train, test_set, y_test = split_dataset(neg + pos, targets)

    train_set = PolarityDataset(train_set, y_train)
    test_set = PolarityDataset(test_set, y_test)

    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE_TRANSFORMER_POL, shuffle=True)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE_TRANSFORMER_POL)

    model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment",
            num_labels=1,
            ignore_mismatched_sizes=True
        ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR_TRANSFORMER)

    best_model = None
    best_test_acc = 0
    for i in range(EPOCHS_TRANSFORMER):
        start = time.time()
        new_best = False

        loss_tr, acc_tr, f1_tr = train(model, optimizer, train_dl)
        loss_test, acc_test, f1_test = evaluate(model, test_dl)

        # save new best model if curr test accuracy is higher than
        # the best accuracy so far
        if acc_test > best_test_acc:
            best_model = copy.deepcopy(model)
            best_test_acc = acc_test
            new_best = True

        print(f"Epoch {i+1}")
        print(f"\tTrain Loss: {loss_tr:.3f}\tTrain Acc: {acc_tr:.3f}\tTrain F1: {f1_tr:.3f}")
        print(f"\tTest Loss: {loss_test:.3f}\tTest Acc: {acc_test:.3f}\tTest F1: {f1_test:.3f}")
        print(f"\tElapsed: {time.time() - start:.3f}")
        if new_best: print("\tNew Best Model")
        
        print("")

    loss_ts, acc_ts, f1_ts = evaluate(best_model, test_dl)
    print(f"Best weights")
    print(f"Loss: {loss_ts:.3f} - Acc: {acc_ts:.3f} - F1: {f1_ts:.3f}")

    # save best weights
    if SAVE:
        make(SAVE_PATH_TRANSFORMER)
        best_model.save_pretrained(os.path.join(SAVE_PATH_TRANSFORMER, f"pol_{FOLD_N}"))
        print("Weights saved")


if __name__ == "__main__":
    main()
import copy
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.corpus import subjectivity
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from dataset import SubjectivityDataset
from settings import BATCH_SIZE_TRANSFORMER_SUBJ, DEVICE, EPOCHS_TRANSFORMER, FOLD_N, LR_TRANSFORMER, SAVE, SAVE_PATH_TRANSFORMER
from train_utils import evaluate, train
from utils import split_dataset, make


def main():
    # Get data
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')

    #? Reduces the size of the dataset when running on CPU.
    #? Less comuputational requirements during test
    if DEVICE != "cuda":
        obj = obj[:20]
        subj = subj[:20]
        print("[Warning] Cuda not detected, a subset of the dataset will be used.")

    # Compute lebels and split in train/test set
    labels = [0] * len(obj) + [1] * len(subj)
    train_set, y_train, test_set, y_test = split_dataset(obj + subj, labels)

    train_set = SubjectivityDataset(train_set, y_train)
    test_set = SubjectivityDataset(test_set, y_test)

    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE_TRANSFORMER_SUBJ, shuffle=True)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE_TRANSFORMER_SUBJ)

    model = AutoModelForSequenceClassification.from_pretrained(
                "cffl/bert-base-styleclassification-subjective-neutral", 
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
        best_model.save_pretrained(os.path.join(SAVE_PATH_TRANSFORMER, f"subj_{FOLD_N}"))
        print("Weights saved")


if __name__ == "__main__":
    main()
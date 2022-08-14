"""
The purpose of this file is to train a subjectivity classifier using GRU.
"""
import copy
import os
import pickle
import sys
import time
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.corpus import subjectivity
from torch import optim
from torch.utils.data import DataLoader

from dataset import SubjectivityDataset
from model import GRUAttention
from settings import SAVE, SAVE_PATH_GRU, WEIGHT_DECAY, BATCH_SIZE_GRU_SUBJ, EPOCHS_GRU, FOLD_N, DEVICE, LR_GRU
from train_utils import evaluate, train
from utils import collate_fn, make, make_w2id, split_dataset, init_weights


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
    X_train, y_train, X_test, y_test = split_dataset(obj + subj, labels)

    # Compute polarity w2id
    vocab = set([w for d in X_train for w in d])
    w2id, w2id_size = make_w2id(vocab)

    train_set = SubjectivityDataset(X_train, y_train, w2id)
    test_set = SubjectivityDataset(X_test, y_test, w2id)
    
    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE_GRU_SUBJ, collate_fn=collate_fn, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE_GRU_SUBJ   , collate_fn=collate_fn)

    model = GRUAttention(num_embeddings=w2id_size).to(DEVICE)
    model.apply(init_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LR_GRU, weight_decay=WEIGHT_DECAY)

    best_model = None
    best_test_acc = 0
    for i in range(EPOCHS_GRU):
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

    # save best weights and w2id
    if SAVE:
        make(SAVE_PATH_GRU)
        torch.save(best_model.state_dict(), os.path.join(SAVE_PATH_GRU, f"subj_cls_{FOLD_N}.pth"))
        with open(os.path.join(SAVE_PATH_GRU, f"subj_w2id_{FOLD_N}.pkl"), 'wb') as f:
            pickle.dump(w2id, f)
        print("Weights saved")


if __name__ == "__main__":
    main()
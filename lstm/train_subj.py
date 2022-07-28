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
from model import LSTMAttention
from settings import SAVE, SAVE_PATH_LSTM, WEIGHT_DECAY, BATCH_SIZE_LSTM_SUBJ, EPOCHS_LSTM, DEVICE, LR_LSTM
from train_utils import evaluate, train
from utils import collate_fn, make, make_w2id, split_dataset, init_weights


def main():
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')
    if DEVICE != "cuda":
        #? Reduces the size of the dataset when running on CPU.
        #? Less comuputational requirements during test
        obj = obj[:20]
        subj = subj[:20]
        print("[Warning] Cuda not detected, a subset of the dataset will be used.")

    labels = [0] * len(obj) + [1] * len(subj)
    X_train, y_train, X_test, y_test = split_dataset(obj + subj, labels)

    vocab = set([w for d in X_train for w in d])
    w2id, w2id_size = make_w2id(vocab)

    train_set = SubjectivityDataset(X_train, y_train, w2id)
    test_set = SubjectivityDataset(X_test, y_test, w2id)
    
    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE_LSTM_SUBJ, collate_fn=collate_fn, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_set, batch_size=BATCH_SIZE_LSTM_SUBJ   , collate_fn=collate_fn)

    model = LSTMAttention(num_embeddings=w2id_size).to(DEVICE)
    model.apply(init_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LR_LSTM, weight_decay=WEIGHT_DECAY)

    best_model = None
    best_test_acc = 0
    for i in range(EPOCHS_LSTM):
        start = time.time()
        new_best = False

        loss_tr, acc_tr = train(model, optimizer, train_dl)
        loss_test, acc_test = evaluate(model, test_dl)

        if acc_test > best_test_acc:
            best_model = copy.deepcopy(model)
            best_test_acc = acc_test
            new_best = True

        print(f"Epoch {i+1}")
        print(f"\tTrain Loss: {loss_tr:.3f}\tTrain Acc: {acc_tr:.3f}")
        print(f"\tTest Loss: {loss_test:.3f}\tTest Acc: {acc_test:.3f}")
        print(f"\tElapsed: {time.time() - start:.3f}")
        if new_best: print("\tNew Best Model")
        
        print("")

    loss_ts, acc_ts = evaluate(best_model, test_dl)
    print(f"Best weights")
    print(f"Loss: {loss_ts:.3f} - Acc: {acc_ts:.3f}")

    if SAVE:
        make(SAVE_PATH_LSTM)
        torch.save(best_model.state_dict(), os.path.join(SAVE_PATH_LSTM, "subj_cls.pth"))
        with open(os.path.join(SAVE_PATH_LSTM, "subj_w2id.pkl"), 'wb') as f:
            pickle.dump(w2id, f)
        print("Weights saved")

if __name__ == "__main__":
    main()
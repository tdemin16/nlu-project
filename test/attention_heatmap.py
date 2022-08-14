"""
ref: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random as rand
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import movie_reviews, subjectivity
from torch.utils.data import DataLoader

from gru.dataset import PolarityDataset, SubjectivityDataset
from gru.model import GRUAttention
from settings import DEVICE, SAVE_PATH_GRU, TEST_DIR
from utils import collate_fn


REVIEWS = os.path.join(TEST_DIR, "reviews.txt") # name of the reviews in the directory TEST_DIR


def make_heatmap_1d(att, words):
    fig, ax = plt.subplots()
    im = ax.imshow(att, cmap=plt.get_cmap("magma"))

    cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal", cmap=plt.get_cmap("magma"), pad=0.7)

    ax.set_xticks(np.arange(len(words)), labels=words)
    ax.axes.yaxis.set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title("Attention for subjectivity detection")
    fig.tight_layout()
    fig.set_figheight(2)

    plt.show()


def make_heatmap_2d(att_1, s_1, att_2, s_2):
    fig, ax = plt.subplots(2)
    fig.suptitle("Attention for polarity classification")
    ax[0].set_title("Attention for an objective sentence")
    ax[1].set_title("Attention for a subjective sentence")

    min_ = min(att_1.min(), att_2.min())
    max_ = max(att_1.max(), att_2.max())
    
    im0 = ax[0].imshow(att_1, vmin=min_, vmax=max_, cmap=plt.get_cmap("magma"))
    im1 = ax[1].imshow(att_2, vmin=min_, vmax=max_, cmap=plt.get_cmap("magma"))

    cbar = ax[0].figure.colorbar(im0, ax=ax, cmap=plt.get_cmap("magma"), pad=-0.45)

    ax[0].set_xticks(np.arange(len(s_1)), labels=s_1)
    ax[1].set_xticks(np.arange(len(s_2)), labels=s_2)

    ax[0].axes.yaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.tight_layout()
    plt.show()


def subjectivity_heatmap():
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')
    
    corpus = obj+subj
    i = rand.randint(0, len(corpus)-1)
    sent = corpus[i]
    
    with open(os.path.join(SAVE_PATH_GRU, "subj_w2id.pkl"), "rb") as fp:
        subj_w2id = pickle.load(fp)
    subj_det = GRUAttention(num_embeddings=len(subj_w2id.keys()))
    subj_det.load_state_dict(torch.load(
        os.path.join(SAVE_PATH_GRU, "subj_cls.pth"), map_location=DEVICE
    ))
    subj_det.to(DEVICE)
    subj_det.eval()

    ds = SubjectivityDataset([sent], [0], subj_w2id)
    dl = DataLoader(ds, collate_fn=collate_fn)
    it = dl.__iter__()
    x, _, l = it.__next__()

    x.to(DEVICE)

    att = subj_det.get_attention_heatmap(x, l)
    att = att.detach().cpu().numpy()
    make_heatmap_1d(att, sent)


def polarity_heatmap():
    reviews = []
    with open(REVIEWS, 'r') as fp:
        for line in fp:
            # remove '\n'
            sentences = line[:-1].lower()
            # apply tokenization in order to reuse training code
            sentences = [word_tokenize(s) for s in sent_tokenize(sentences)]
            reviews.append(sentences)

    doc = reviews[1]
    
    with open(os.path.join(SAVE_PATH_GRU, "pol_w2id.pkl"), "rb") as fp:
        pol_w2id = pickle.load(fp)
    pol_cls = GRUAttention(num_embeddings=len(pol_w2id.keys()))
    pol_cls.load_state_dict(torch.load(
        os.path.join(SAVE_PATH_GRU, "pol_cls.pth"), map_location=DEVICE
    ))
    pol_cls.to(DEVICE)
    pol_cls.eval()

    ds = PolarityDataset([doc], [0], pol_w2id)
    dl = DataLoader(ds, collate_fn=collate_fn)
    it = dl.__iter__()
    x, _, l = it.__next__()

    x.to(DEVICE)

    att = pol_cls.get_attention_heatmap(x, l).squeeze(-1)
    att = att.detach().cpu().numpy()
    
    att_1 = att[:, :len(doc[0])]
    s_1 = doc[0]
    
    start = 0
    for i, x in enumerate(doc):
        if i == 3: break
        start += len(x)

    att_2 = att[:, start:start+len(doc[3])]
    s_2 = doc[3]

    make_heatmap_2d(att_1, s_1, att_2, s_2)


def main():
    subjectivity_heatmap()
    polarity_heatmap()


if __name__ == "__main__":
    main()
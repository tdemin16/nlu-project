import joblib
import numpy as np
import os

from nltk.corpus import movie_reviews, subjectivity

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate, StratifiedKFold

from utils import list2str, lol2str, remove_objective_sents, make
from settings import FILTER, N_SPLITS, RANDOM_STATE, SAVE_PATH_BASELINE


def train_subjectivity_classifier():
    """
    0.92 +-0.01 f1
    """
    # init classifier and vectorizer for Polairty classification
    vectorizer = CountVectorizer()
    eval_classifier = MultinomialNB()

    # get data
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')
    
    # build dataset
    corpus = [list2str(d) for d in obj] + [list2str(d) for d in subj]
    vectors = vectorizer.fit_transform(corpus)
    targets = [0] * len(obj) + [1] * len(subj)

    # train and evaluate
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(eval_classifier, vectors, targets, cv=cv, scoring=['accuracy'])
    scores = np.array(scores['test_accuracy'])

    classifier = MultinomialNB()
    classifier.fit(vectors, targets)

    return scores, classifier, vectorizer


def train_polarity_classifier(subj_classifier, subj_vectorizer):
    """
    With filtering objective phrases: 0.84 +-0.03 f1
    Withoud filtering objective phrases: 0.81 +-0.03 f1
    """
    # init classifier and vectorizer for Polairty classification
    vectorizer = CountVectorizer()
    eval_classifier = MultinomialNB()

    # get data
    neg = movie_reviews.paras(categories='neg')
    pos = movie_reviews.paras(categories='pos')

    # filter phrases
    if FILTER:
        corpus = []
        for d in neg + pos:
            corpus.append(remove_objective_sents(subj_classifier, subj_vectorizer, d))
    else:
        corpus = neg + pos
        corpus = [lol2str(d) for d in corpus]
    
    # build dataset
    vectors = vectorizer.fit_transform(corpus)
    targets = [0] * len(neg) + [1] * len(pos)

    # train and evaluate
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(eval_classifier, vectors, targets, cv=cv, scoring=['accuracy'])
    scores = np.array(scores['test_accuracy'])

    classifier = MultinomialNB()
    classifier.fit(vectors, targets)

    return scores, classifier, vectorizer


def main():
    subj_scores, subj_classifier, subj_vectorizer = train_subjectivity_classifier()
    print(f"Naive Bayes F1 score on subjectivity classification:\n\tAverage: {subj_scores.mean():.3f}\n\tSTD: {subj_scores.std():.3f}")

    pol_scores, pol_classifier, pol_vectorizer = train_polarity_classifier(subj_classifier, subj_vectorizer)
    print(f"Naive Bayes F1 score on polarity classification:\n\tAverage: {pol_scores.mean():.3f}\n\tSTD: {pol_scores.std():.3f}")

    make(SAVE_PATH_BASELINE)
    joblib.dump(subj_classifier, os.path.join(SAVE_PATH_BASELINE, 'subj_cls.joblib'))
    joblib.dump(subj_vectorizer, os.path.join(SAVE_PATH_BASELINE, 'subj_vec.joblib'))
    joblib.dump(pol_classifier, os.path.join(SAVE_PATH_BASELINE, 'pol_cls.joblib'))
    joblib.dump(pol_vectorizer, os.path.join(SAVE_PATH_BASELINE, 'pol_vec.joblib'))


if __name__ == "__main__":
    main()
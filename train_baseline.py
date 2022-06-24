import numpy as np

from nltk.corpus import movie_reviews, subjectivity

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate, StratifiedKFold

from utils import lol2str, list2str


def train_subjectivity_classifier():
    # init classifier and vectorizer for Polairty classification
    vectorizer = CountVectorizer()
    classifier = MultinomialNB()

    # get data
    obj = subjectivity.sents(categories='obj')
    subj = subjectivity.sents(categories='subj')
    
    # build dataset
    corpus = [list2str(d) for d in obj] + [list2str(d) for d in subj]
    vectors = vectorizer.fit_transform(corpus)
    targets = [-1] * len(obj) + [1] * len(subj)

    # train and evaluate
    scores = cross_validate(classifier, vectors, targets, cv=StratifiedKFold(n_splits=10), scoring=['f1_micro'])
    scores = np.array(scores['test_f1_micro'])

    return scores


def train_polarity_classifier():
    # init classifier and vectorizer for Polairty classification
    vectorizer = CountVectorizer()
    classifier = MultinomialNB()

    # get data
    neg = movie_reviews.paras(categories='neg')
    pos = movie_reviews.paras(categories='pos')
    
    # build dataset
    corpus = [lol2str(d) for d in neg] + [lol2str(d) for d in pos]
    vectors = vectorizer.fit_transform(corpus)
    targets = [-1] * len(neg) + [1] * len(pos)

    # train and evaluate
    scores = cross_validate(classifier, vectors, targets, cv=StratifiedKFold(n_splits=10), scoring=['f1_micro'])
    scores = np.array(scores['test_f1_micro'])

    return scores


def main():
    subj_scores = train_subjectivity_classifier()
    print(f"Naive Bayes F1 score on subjectivity classification:\n\tAverage: {subj_scores.mean():.2f}\n\tSTD: {subj_scores.std():.2f}")

    pol_scores = train_polarity_classifier()
    print(f"Naive Bayes F1 score on polarity classification:\n\tAverage: {pol_scores.mean():.2f}\n\tSTD: {pol_scores.std():.2f}")

if __name__ == "__main__":
    main()
import numpy as np

from nltk.corpus import movie_reviews, subjectivity

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate, StratifiedKFold

from utils import lol2str


def main():
    pol_vectorizer = CountVectorizer()
    pol_classifier = MultinomialNB()

    neg = movie_reviews.paras(categories='neg')
    pos = movie_reviews.paras(categories='pos')
    
    pol_corpus = [lol2str(d) for d in neg] + [lol2str(d) for d in pos]
    pol_vectors = pol_vectorizer.fit_transform(pol_corpus)
    pol_targets = [-1] * len(neg) + [1] * len(pos)

    pol_scores = np.array(cross_validate(pol_classifier, pol_vectors, pol_targets, cv=StratifiedKFold(n_splits=10), scoring=['f1_micro'])['test_f1_micro'])
    print(f"Naive Bayes F1 score:\n\tAverage: {pol_scores.mean():.2f}\n\tSTD: {pol_scores.std():.2f}")

if __name__ == "__main__":
    main()
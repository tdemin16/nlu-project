import os

def list2str(l):
    """
    Returns a string from a list
    """
    return ' '.join(w for w in l)


def remove_objective_sents(classifier, vectorizer, document):
    """
    Remove the objective sentences from a document and returns the filtered document.
    """
    document = [list2str(p) for p in document]
    vectors = vectorizer.transform(document)
    estimated_subj = classifier.predict(vectors)
    filt_sent = [d for d, est in zip(document, estimated_subj) if est == 1]
    filt_doc = list2str(filt_sent)
    return filt_doc


def make(path):
    if not os.path.exists(path):
        os.makedirs(path)
def lol2str(doc):
    """
    List of lists to string.
    """
    return " ".join([w for sent in doc for w in sent])

def list2str(l):
    """
    Returns a string from a list
    """
    return ' '.join(w for w in l)
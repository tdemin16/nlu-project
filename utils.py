def lol2str(doc):
    """
    List of lists to string.
    """
    return " ".join([w for sent in doc for w in sent])
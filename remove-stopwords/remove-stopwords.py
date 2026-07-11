def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    non_stopwords=[]
    for token in tokens:
        if token not in stopwords:
            non_stopwords.append(token)
    return non_stopwords
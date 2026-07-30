import numpy as np
def get_frequent(word,words):
    cnt=0
    for w in words:
        if w==word:
            cnt+=1
    return cnt
def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    vec=[]
    for word in vocab:
        vec.append(get_frequent(word,tokens))
    return np.array(vec,dtype=int)
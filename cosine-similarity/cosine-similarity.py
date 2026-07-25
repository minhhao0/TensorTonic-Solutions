import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a=np.array(a)
    b=np.array(b)
    numerator=(a*b).sum()
    n_a=np.linalg.norm(a)
    n_b=np.linalg.norm(b)
    if n_a==0 or n_b==0:
        return 0
    denominator=n_a*n_b
    return numerator/denominator
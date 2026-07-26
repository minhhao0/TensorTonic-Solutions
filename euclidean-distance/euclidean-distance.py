import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x=np.array(x)
    y=np.array(y)
    tmp=(x-y)*(x-y)
    s=tmp.sum()
    return np.sqrt(s)
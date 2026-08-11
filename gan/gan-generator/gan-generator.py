import numpy as np

def generator(z, W, b):
    """
    Returns: np.ndarray of shape (batch, output_dim) with tanh-activated values rounded to 4 decimals
    """
    z=np.array(z)
    W=np.array(W)
    b=np.array(b)
    return np.tanh(z@W+b)
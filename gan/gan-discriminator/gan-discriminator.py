import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def discriminator(x, W):
    """
    Returns: np.ndarray of shape (batch, 1) with probabilities rounded to 4 decimals
    """
    out=[]
    x=np.array(x)
    W=np.array(W)
    batches=x.shape[0]
    for i in range(batches):
        batch=x[i]
        out.append(sigmoid(batch@W))
    return np.array(out)
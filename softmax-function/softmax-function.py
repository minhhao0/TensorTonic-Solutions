import numpy as np
def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x=np.array(x)
    denorminator=np.sum(np.exp(x-np.max(x,keepdims=True)),axis=-1,keepdims=True)
    numerator=np.exp(x-np.max(x,keepdims=True))
    return numerator/denorminator
import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    # YOUR CODE HERE
    # Step 1: First linear transformation and activation
    W1=np.array(W1)
    W2=np.array(W2)
    x=np.array(x)
    h=np.maximum(x@W1.T,0)
    # Step 2: Second linear transformation (no activation)
    f_x=h@W2.T
    # Step 3: Skip addition and final activation
    y=np.maximum(f_x+x,0)
    return y
    

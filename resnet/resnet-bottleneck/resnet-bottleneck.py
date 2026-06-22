import numpy as np
def relu(x):
    return np.maximum(x,0)
def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    # YOUR CODE HERE
    x=np.array(x)
    W1=np.array(W1)
    W2=np.array(W2)
    W3=np.array(W3)
    if Ws is None:
        s=x
    else:
        Ws=np.array(Ws)
        s=x@Ws
    y1=relu(x@W1)
    y2=relu(y1@W2)
    y3=y2@W3
    y=relu(y3+s)
    return y
    
    
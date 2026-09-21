import numpy as np
def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    # Write code here
    X=np.array(X)
    W=np.array(W)
    b=np.array(b)
    return (X@W+b).tolist()
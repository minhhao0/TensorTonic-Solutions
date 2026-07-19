import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # Write code here
    X=np.array(X)
    min=X.min(axis=axis,keepdims=True)
    max=X.max(axis=axis,keepdims=True)
    denominator=max-min
    numerator=X-min
    denominator=np.maximum(denominator,eps)
    new=numerator/denominator
    return new
    
import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    X=np.array(X)
    N=X.shape[0]
    X_c=X-X.mean(axis=0,keepdims=True)
    cor=(X_c.T@X_c)/(N-1)
    return cor
    
    
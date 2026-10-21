import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    # Write code here
    X=np.array(X)
    y=np.array(y)
    invert=np.linalg.inv(X.T@X)
    s=X.T@y
    result=invert@s
    return result.tolist()
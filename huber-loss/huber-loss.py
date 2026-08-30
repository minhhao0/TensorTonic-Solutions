import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    error=y_true-y_pred
    result=np.where(np.abs(error)>delta,delta*(np.abs(error)-0.5*delta),0.5*(error**2))
    return float(result.mean())
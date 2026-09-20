import numpy as np

def auc(fpr: list, tpr: list) -> float:
    """
    Returns the area as a float.
    """
    # Write code here
    M=len(fpr)
    auc=0
    for i in range(M-1):
        auc+=0.5*(fpr[i+1]-fpr[i])*(tpr[i]+tpr[i+1])
    return auc
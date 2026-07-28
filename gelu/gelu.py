import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x=np.array(x)
    erf=np.vectorize(math.erf)
    y=0.5*x*(1+erf(x/math.sqrt(2)))
    return y
    

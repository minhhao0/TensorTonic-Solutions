import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sw(x):
    return x*sigmoid(x)
def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    vectorize=np.vectorize(sw)
    return vectorize(x)
    
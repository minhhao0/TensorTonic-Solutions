import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x=np.array(x)
    numerator=np.exp(x)- np.exp(-x)
    denorminator=np.exp(x)+np.exp(-x)
    return numerator/denorminator
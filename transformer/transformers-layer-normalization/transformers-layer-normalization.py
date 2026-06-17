import numpy as np
import math

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    # result=[]
    # for i in range(x.shape[0]):
    #     x_i=x[i]
    #     mean=x_i.mean()
    #     std=x_i.std()
        
    #     result_i=gamma*((x_i-mean)/(math.sqrt(std**2+eps)))+beta
    #     result.append(result_i)
    mean=np.mean(x,axis=-1,keepdims=True)
    variance=np.var(x,axis=-1,keepdims=True)
    x=x-mean
    x=x/np.sqrt(variance+eps)
    x=gamma*x+beta
    print(x)
    return x
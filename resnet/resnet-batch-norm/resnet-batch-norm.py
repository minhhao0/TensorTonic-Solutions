import numpy as np

def batch_norm(x,gamma,beta,epsilon=1e-5):
    mean=np.mean(x,axis=0,keepdims=True)
    variance=np.var(x,axis=0,keepdims=True)
    x=x-mean
    x=x/(np.sqrt(variance+epsilon))
    x=gamma*x+beta
    return x
def relu(x):
    return np.maximum(x,0)
def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    # YOUR CODE HERE
    x=np.array(x)
    W1=np.array(W1)
    W2=np.array(W2)
    gamma1=np.array(gamma1)
    beta1=np.array(beta1)
    gamma2=np.array(gamma2)
    beta2=np.array(beta2)
    if mode=='post':
        y1=x@W1
        y1=batch_norm(y1,gamma1,beta1)
        y1=relu(y1)
        y2=y1@W2
        y2=batch_norm(y2,gamma2,beta2)
        y=y2+x
        y=relu(y)
        return {'output':y,'mode':mode}
    elif mode=='pre':
        x1=batch_norm(x,gamma1,beta1)
        x1=relu(x1)
        y1=x1@W1
        y1=batch_norm(y1,gamma2,beta2)
        y1=relu(y1)
        y2=y1@W2
        y=y2+x
        return {'output':y,'mode':mode}
import numpy as np

def get_alpha_bar(betas):
    """
    Compute cumulative product of (1 - beta).
    Returns list of floats rounded to 6 decimals.
    """
    # YOUR CODE HERE
    T=len(betas)
    alpha_bar=[]
    pro=1
    for i in range(T):
        pro*=(1-betas[i])
        alpha_bar.append(pro)
    return alpha_bar
    

def forward_diffusion(x_0, t, betas, epsilon):
    """
    Returns: tuple of (np.ndarray x_t, np.ndarray epsilon) with same shape as x_0
    """
    # YOUR CODE HERE
    alpha_bar=get_alpha_bar(betas)
    x_0=np.array(x_0)
    betas=np.array(betas)
    epsilon=np.array(epsilon)
    alpha_bar=np.array(alpha_bar)
    alpha_t=alpha_bar[t-1]
    x_t=np.sqrt(alpha_t)*x_0+np.sqrt(1-alpha_t)*epsilon
    return x_t
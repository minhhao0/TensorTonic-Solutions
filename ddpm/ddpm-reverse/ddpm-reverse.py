import numpy as np
def get_alpha_bar(betas):
    result=[]
    product=1
    for beta in betas:
        alpha_t=1-beta
        product*=alpha_t
        result.append(product)
    return result
def get_mean(betas,epsilon_pred,x_t,t):
    alpha_bars=get_alpha_bar(betas)
    mean_t=(1/np.sqrt(1-betas[t-1]))*(x_t-(betas[t-1]/np.sqrt(1-alpha_bars[t-1]))*epsilon_pred)
    return mean_t
def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    # YOUR CODE HERE
    x_t=np.array(x_t)
    epsilon_pred=np.array(epsilon_pred)
    means=get_mean(betas,epsilon_pred,x_t,t)
    z=np.array(z)
    if t==1:
        return means
    return means+np.sqrt(betas[t-1])*z
    
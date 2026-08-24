import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Returns: float scalar KL divergence averaged over the batch
    """
    # Your implementation here
    mu=np.array(mu)
    log_var=np.array(log_var)
    var=np.exp(log_var)
    print(var)
    print(np.exp(log_var))
    kl_score=-0.5*(1+log_var-mu*mu-var).sum(axis=1,keepdims=True).mean()
    return kl_score

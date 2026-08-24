import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns: dict with "total", "recon", and "kl" loss values as floats
    """
    # Your implementation here
    x=np.array(x)
    x_recon=np.array(x_recon)
    mu=np.array(mu)
    log_var=np.array(log_var)
    recon=((x-x_recon)*(x-x_recon)).sum(axis=1,keepdims=True).mean()
    kl=-0.5*(1+log_var-mu*mu-np.exp(log_var)).sum(axis=-1,keepdims=True).mean()
    return {
        "total":recon+kl,
        "recon":recon,
        "kl":kl
    }
    

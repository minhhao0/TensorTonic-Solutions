import numpy as np

def vae_encoder(x: np.ndarray, W_mu: np.ndarray, b_mu: np.ndarray, W_logvar: np.ndarray, b_logvar: np.ndarray) -> dict:
    """
    Returns: dict with 'mu' and 'log_var' as np.ndarrays of shape (batch, latent_dim)
    """
    # Your implementation here
    x=np.array(x)
    W_mu=np.array(W_mu)
    b_mu=np.array(b_mu)
    W_logvar=np.array(W_logvar)
    b_logvar=np.array(b_logvar)
    mu=x@W_mu+b_mu
    logvar=x@W_logvar+b_logvar
    return mu,logvar
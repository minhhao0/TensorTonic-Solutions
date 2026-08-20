import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray z of shape (batch, latent_dim) sampled via reparameterization
    """
    # Your implementation here
    mu=np.array(mu)
    log_var=np.array(log_var)
    epsilon=np.array(epsilon)
    return mu+np.exp(0.5*log_var)*epsilon

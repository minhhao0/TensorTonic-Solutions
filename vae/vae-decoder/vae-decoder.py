import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def vae_decoder(z: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray of shape (batch, output_dim) with reconstructed data
    """
    # Your implementation here
    z=np.array(z)
    W_dec=np.array(W_dec)
    b_dec=np.array(b_dec)
    return z@W_dec+b_dec

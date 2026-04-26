import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    """Compute input gate and candidate memory."""
    # YOUR CODE HERE
    concatenate=np.concatenate([h_prev,x_t],axis=-1).T
    i_t=sigmoid(W_i@concatenate+b_i)
    c_tilde=np.tanh(W_c@concatenate+b_c)
    return (i_t.T,c_tilde.T)
import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    h=[h_0]
    for t in range(1,X.shape[1]):
            h_t=np.tanh(h[t-1]@W_hh+X[:,t,:]@W_xh.T+b_h)
            h.append(h_t)
    return np.stack(h,axis=1),h[-1]         
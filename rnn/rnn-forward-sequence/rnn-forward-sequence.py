import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    h_final=None
    hidden_states=[h_0]
    # YOUR CODE HERE
    print(X.shape)
    for x in range(1,X.shape[1]+1):
        x_t=X[:,x-1,:]
        h_prev=hidden_states[x-1]
        h_t=np.tanh(x_t@W_xh.T+h_prev@W_hh.T+b_h)
        hidden_states.append(h_t)
    h_final=hidden_states[-1]
    hidden_states=hidden_states[1:]
    return np.stack(hidden_states,axis=1),h_final
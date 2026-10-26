import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def update_gate(h_prev: np.ndarray, x_t: np.ndarray,
          W_z: np.ndarray, b_z: np.ndarray) -> np.ndarray:
    """
    Returns the float64 update-gate values.
    """
    h_prev=np.array(h_prev)
    x_t=np.array(x_t)
    W_z=np.array(W_z)
    b_z=np.array(b_z)
    print(x_t.shape)
    print(h_prev.shape)
    h=np.concatenate([h_prev,x_t],axis=-1)
    z_t=sigmoid(h@W_z.T+b_z)
    return z_t
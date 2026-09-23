import numpy as np

def rnn_step_forward(x_t: list, h_prev: list, Wx: list, Wh: list, b: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (H,).
    """
    # Write code here
    x_t=np.array(x_t)
    h_prev=np.array(h_prev)
    Wx=np.array(Wx)
    Wh=np.array(Wh)
    b=np.array(b)
    a_t=x_t@Wx+h_prev@Wh+b
    h_t=np.tanh(a_t)
    return h_t
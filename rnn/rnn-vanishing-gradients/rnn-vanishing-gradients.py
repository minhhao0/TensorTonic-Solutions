import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    result=[1]
    for i in range(1,T):
        result.append(result[-1]*np.linalg.matrix_norm(W_hh,ord=2))
    return result
        
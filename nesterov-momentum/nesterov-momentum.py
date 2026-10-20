import numpy as np

def nesterov_momentum_step(w: list, v: list, grad: list, lr: float = 0.01, momentum: float = 0.9) -> dict:
    """
    Returns a dictionary with new_w and new_v.
    """
    # Write code here
    new_w=[]
    new_v=[]
    for i in range(len(v)):
        v_t=momentum*v[i]+lr*grad[i]
        new_v.append(v_t)
        new_w.append(w[i]-v_t)
    return {
        "new_w":np.array(new_w),
        "new_v":np.array(new_v)
    }
    
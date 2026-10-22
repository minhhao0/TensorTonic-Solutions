import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    new_s=[]
    new_w=[]
    T=len(s)
    s=np.array(s)
    w=np.array(w)
    g=np.array(g)
    for i in range(T):
        st=s[i]*beta+(1-beta)*(g[i]**2)
        wt=w[i]-(lr/np.sqrt(st+eps))*g[i]
        new_s.append(st)
        new_w.append(wt)
    return new_w,new_s
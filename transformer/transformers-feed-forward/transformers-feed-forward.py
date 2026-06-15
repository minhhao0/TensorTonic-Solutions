import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here
    batch=x.shape[0]
    result=[]
    for i in range(batch):
        x_i=x[i,:,:]
        h=x_i@W1+b1
        print(h.shape)
        a=np.maximum(h,0)
        print(a.shape)
        ffn_x=a@W2+b2
        result.append(ffn_x)
    return np.stack(result)
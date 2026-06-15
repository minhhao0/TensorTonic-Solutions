import numpy as np
import math
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    d_model=W_q.shape[0]
    d_k=d_model/num_heads
    batch=Q.shape[0]
    result=[]
    for i in range(batch):
        q=Q[i,:,:]
        k=K[i,:,:]
        v=V[i,:,:]
        q=q@W_q
        k=k@W_k
        v=v@W_v
        vs=np.hsplit(v,num_heads)
        ks=np.hsplit(k,num_heads)
        qs=np.hsplit(q,num_heads)
        head=[]
        for i in range(num_heads):
            s_i=qs[i]@ks[i].T
            #normalize
            s_i=s_i/math.sqrt(d_k)
            s_i=softmax(s_i)
            #calculate head i
            h_i=s_i@vs[i]
            head.append(h_i)
        output=np.concatenate(head,axis=1)
        output=output@W_o
        output=np.expand_dims(output,axis=0)
        result.append(output)
    return np.concatenate(result)
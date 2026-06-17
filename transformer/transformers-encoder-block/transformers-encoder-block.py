import numpy as np
import math

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mean=np.mean(x,axis=-1,keepdims=True)
    variance=np.var(x,axis=-1,keepdims=True)
    x=x-mean
    x=x/np.sqrt(variance+eps)
    x=gamma*x+beta
    return x

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
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

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    batch=x.shape[0]
    result=[]
    for i in range(batch):
        x_i=x[i,:,:]
        z=x_i@W1+b1
        a=np.maximum(z,0)
        a=a@W2+b2
        result.append(a)
    return np.stack(result)

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    #multihead attention
    mha=multi_head_attention(x,x,x,W_q,W_k,W_v,W_o,num_heads)
    #residual addition
    x=x+mha
    x=layer_norm(x,gamma1,beta1) 
    ffn_x=feed_forward(x, W1, b1,W2, b2)
    x=x+ffn_x
    output=layer_norm(x,gamma2,beta2)
    return output
import numpy as np
def softmax(x,axis=-1):
    e_x=np.exp(x-np.max(x,axis=axis,keepdims=True))
    return e_x/np.sum(e_x,axis=axis,keepdims=True)
def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.pow(x,3))))
def layernorm(x,gamma=1,beta=0,epsilon=1e-6):
    mean=np.mean(x,axis=-1,keepdims=True)
    variance=np.var(x,axis=-1,keepdims=True)
    x=gamma*(x-mean)/np.sqrt(variance+epsilon)+beta
    return x
def mlp(x,W1,W2):
    z=x@W1
    y=gelu(z)@W2
    return y
def msa(x,num_heads,Wq,Wk,Wv,Wo):
    Q=x@Wq
    K=x@Wk
    V=x@Wv
    d_k=x.shape[1]/num_heads
    Qs=np.hsplit(Q,num_heads)
    Ks=np.hsplit(K,num_heads)
    Vs=np.hsplit(V,num_heads)
    h=[]
    for i in range(num_heads):
        q=Qs[i]
        k=Ks[i]
        v=Vs[i]
        s=q@k.T
        s=s/np.sqrt(d_k)
        s=softmax(s)
        s=s@v
        h.append(s)
    output=np.concatenate(h,axis=-1)
    output=output@Wo
    return output
def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                      Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
    """
    ViT Transformer encoder block with Pre-LayerNorm.
    Weight matrices are provided as inputs for deterministic testing.
    """
    # YOUR CODE HERE
    x=np.array(x)
    Wq=np.array(Wq)
    Wk=np.array(Wk)
    Wv=np.array(Wv)
    Wo=np.array(Wo)
    W1=np.array(W1)
    W2=np.array(W2)
    batch_size=x.shape[0]
    result=[]
    for i in range(batch_size):
        x_i=x[i]
        #1st layer norm
        x_1=layernorm(x_i)
        #multi head attention
        mha=msa(x_1,num_heads,Wq,Wk,Wv,Wo)
        #1st residual connection
        x_2=x_i+mha
        #2nd layer norm
        x_3=layernorm(x_2)
        #mlp + 2nd residual connection
        x_4=x_2+mlp(x_3,W1,W2)
        x_4=np.expand_dims(x_4,axis=0)
        result.append(x_4)
    return np.concatenate(result)
        
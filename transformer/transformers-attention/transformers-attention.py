import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k=K.shape[2]
    batch=Q.shape[0]
    s_d_k=math.sqrt(d_k)
    result=[]
    softmax=nn.Softmax(dim=1)
    for i in range(batch):
        q=Q[i,:,:]
        k=K[i,:,:]
        v=V[i,:,:]
        s=q@k.T
        s_scaled=torch.div(s,s_d_k)
        w=softmax(s_scaled)
        o=w@v
        result.append(o.unsqueeze(0))
    return torch.cat(result,dim=0)

    
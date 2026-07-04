import torch
import torch.nn as nn
def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor, center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> tuple:
    """
    Returns tuple (W_in_updated, W_out_updated), each the same shape as the inputs, after one SGNS SGD step.
    """
    # YOUR CODE HERE
    sigmoid=nn.Sigmoid()
    W_in=torch.Tensor(W_in)
    W_out=torch.Tensor(W_out)
    v_c=W_in[center_id]
    u_o=W_out[pos_id]
    u_n=[]
    for neg_id in neg_ids:
        u_n.append(W_out[neg_id])
    u_n=torch.stack(u_n)
    #Step 1 Positive score:
    s_o=v_c@u_o.T
    #Step 2 Negative score:
    s_n=v_c@u_n.T
    #Step 3 Center gradients:
    #coefficient positive
    c_p=sigmoid(s_o)-1
    #coefficient negative
    c_n=sigmoid(s_n)
    g_v_c=c_p*u_o+c_n@u_n
    g_u_o=c_p*v_c
    g_u_n=[]
    for i in range(c_n.shape[0]):
        g_u_n.append(c_n[i]*v_c)
    g_u_n=torch.stack(g_u_n)
    v_c=v_c-lr*g_v_c
    u_o=u_o-lr*g_u_o
    #accumulate gradients for the same negative ids
    for i in range(len(neg_ids)):
        W_out[neg_ids[i]]-=lr*g_u_n[i]
    W_in[center_id]=v_c
    W_out[pos_id]=u_o
    
    return W_in,W_out
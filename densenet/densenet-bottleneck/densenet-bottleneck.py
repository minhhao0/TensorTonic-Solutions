import torch
import torch.nn.functional as F
def batch_norm(x,g,b,m,v,eps):
    #broadcast
    x=(x-m[None,:,None,None])/torch.sqrt(v[None,:,None,None]+eps)
    x=g[None,:,None,None]*x+b[None,:,None,None]
    return x
def conv(x,weight,type_="1"):
    if type_=="1":
        z=F.conv2d(x,weight,bias=None,stride=1,padding=0)
    else:
        z=F.conv2d(x,weight,bias=None,stride=1,padding=1)
    return z
def relu(x):
    z=F.relu(x)
    return z
def bottleneck_layer(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, conv1_weight,
                     bn2_gamma, bn2_beta, bn2_mean, bn2_var, conv2_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W) after the two-stage bottleneck composite.
    """
    # YOUR CODE HERE
    x=torch.tensor(x,dtype=torch.float64)
    bn1_gamma=torch.tensor(bn1_gamma,dtype=torch.float64)
    bn1_beta=torch.tensor(bn1_beta,dtype=torch.float64)
    bn1_mean=torch.tensor(bn1_mean,dtype=torch.float64)
    bn1_var=torch.tensor(bn1_var,dtype=torch.float64)
    conv1_weight=torch.tensor(conv1_weight,dtype=torch.float64)
    bn2_gamma=torch.tensor(bn2_gamma,dtype=torch.float64)
    bn2_beta=torch.tensor(bn2_beta,dtype=torch.float64)
    bn2_mean=torch.tensor(bn2_mean,dtype=torch.float64)
    bn2_var=torch.tensor(bn2_var,dtype=torch.float64)
    conv2_weight=torch.tensor(conv2_weight,dtype=torch.float64)
    #stage 1
    x_1=batch_norm(x,bn1_gamma,bn1_beta,bn1_mean,bn1_var,eps)
    #relu
    z_1=relu(x_1)
    #conv1x1
    y1=conv(z_1,conv1_weight)
    #stage 2
    x_2=batch_norm(y1,bn2_gamma,bn2_beta,bn2_mean,bn2_var,eps)
    #relu
    z_2=relu(x_2)
    #conv3x3
    y2=conv(z_2,conv2_weight,type_="3")
    return y2
import torch
import torch.nn.functional as F
def batchnorm(x,g,b,m,v,eps):
    x=(x-m[None,:,None,None])/torch.sqrt(v[None,:,None,None]+eps)
    x=g[None,:,None,None]*x+b[None,:,None,None]
    return x
def relu(x):
    z=F.relu(x)
    return z
def conv(x,conv_weight):
    y=F.conv2d(x,conv_weight,bias=None,stride=1,padding=0)
    return y
def averagepool(x):
    return F.avg_pool2d(x,(2,2),stride=2)
def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, out_channels, H//2, W//2) after BN-ReLU-1x1Conv then 2x2 average pooling.
    """
    # YOUR CODE HERE
    x=torch.tensor(x,dtype=torch.float64)
    bn_gamma=torch.tensor(bn_gamma,dtype=torch.float64)
    bn_beta=torch.tensor(bn_beta,dtype=torch.float64)
    bn_mean=torch.tensor(bn_mean,dtype=torch.float64)
    bn_var=torch.tensor(bn_var,dtype=torch.float64)
    conv_weight=torch.tensor(conv_weight,dtype=torch.float64)
    #batchnorm
    x_n=batchnorm(x,bn_gamma,bn_beta,bn_mean,bn_var,eps)
    #relu
    z=relu(x_n)
    #convolution
    y=conv(z,conv_weight)
    #average pooling
    out=averagepool(y)
    return out

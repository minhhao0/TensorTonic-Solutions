import torch
import torch.nn.functional as F
def batch_norm(x,bn_gamma,bn_beta,bn_mean,bn_var,eps):
    x=(x-bn_mean[None,:,None,None])/torch.sqrt(bn_var[None,:,None,None]+eps)
    x=bn_gamma[None,:,None,None]*x+bn_beta[None,:,None,None]
    return x
def relu(x):
    rl=F.relu(x)
    return rl
def conv3x3(x,conv_weight):
    # return h
    h=F.conv2d(x,conv_weight,bias=None,stride=1,padding=1)
    return h
def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W): BN, ReLU, then a 3x3 same-padding convolution.
    """
    # YOUR CODE HERE
    x=torch.tensor(x,dtype=torch.float64)
    bn_gamma=torch.tensor(bn_gamma,dtype=torch.float64)
    bn_beta=torch.tensor(bn_beta,dtype=torch.float64)
    bn_mean=torch.tensor(bn_mean,dtype=torch.float64)
    bn_var=torch.tensor(bn_var,dtype=torch.float64)
    conv_weight=torch.tensor(conv_weight,dtype=torch.float64)
    n,c,h,w=x.shape
    #batchnorm
    z=batch_norm(x,bn_gamma,bn_beta,bn_mean,bn_var,eps)
    #relu
    z=relu(z)
    #conv 3x3
    y=conv3x3(z,conv_weight)
    return y
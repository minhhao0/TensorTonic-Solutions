import torch
import torch.nn.functional as F
def batchnorm(x,gamma,beta,mean,var,epsilon):
    x=(x-mean[None,:,None,None])/torch.sqrt(var[None,:,None,None]+epsilon)
    x=gamma[None,:,None,None]*x+beta[None,:,None,None]
    return x
def relu(x):
    z=F.relu(x)
    return z
def convd(x,conv_weight):
    y=F.conv2d(x,conv_weight,bias=None,stride=1,padding=1)
    return y
def composite(x,conv_weight,gamma,beta,mean,var,epsilon):
    #batch_norm
    z=batchnorm(x,gamma,beta,mean,var,epsilon)
    #relu
    y=relu(z)
    #convolution
    out=convd(y,conv_weight)
    return out
def dense_block(x, layers, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, C + L*growth_rate, H, W).
    """
    # YOUR CODE HERE
    x=torch.tensor(x,dtype=torch.float64)
    n_layers=len(layers)
    input_=x
    for i in range(n_layers):
        layer=layers[i]
        var=torch.tensor(layer['bn_var'],dtype=torch.float64)
        beta=torch.tensor(layer['bn_beta'],dtype=torch.float64)
        mean=torch.tensor(layer['bn_mean'],dtype=torch.float64)
        gamma=torch.tensor(layer['bn_gamma'],dtype=torch.float64)
        conv_weight=torch.tensor(layer['conv_weight'],dtype=torch.float64)
        out=composite(input_,conv_weight,gamma,beta,mean,var,eps)
        input_=torch.cat((input_,out),1)
    return input_
        
        
        
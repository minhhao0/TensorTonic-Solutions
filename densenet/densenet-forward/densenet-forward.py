import torch
import torch.nn.functional as F
def batch_norm(x,g,b,m,v,eps):
    x=(x-m[None,:,None,None])/torch.sqrt(v[None,:,None,None]+eps)
    x=g[None,:,None,None]*x+b[None,:,None,None]
    return x
def relu(x):
    return F.relu(x)
def conv(x,conv_weight,type_='1x1'):
    if type_=='1x1':
        return F.conv2d(x,conv_weight,bias=None,stride=1,padding=0)
    return F.conv2d(x,conv_weight,bias=None,stride=1,padding=1)    
def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    """
    Returns torch.Tensor: BN-ReLU-3x3Conv (padding 1, no bias) producing growth_rate channels.
    """
    # YOUR CODE HERE
    #batch-norm
    z=batch_norm(x,bn_gamma,bn_beta,bn_mean,bn_var,eps)
    #relu
    y=relu(z)
    #conv3x3
    out=conv(y,conv_weight,'3x3')
    return out
def dense_block(x, layers, eps):
    """
    Returns torch.Tensor: concat of x and every composite-layer output (channels grow by growth_rate per layer).
    """
    # YOUR CODE HERE
    n_layers=len(layers)
    input_=x
    for i in range(n_layers):
        layer=layers[i]
        gamma=torch.tensor(layer['bn_gamma'],dtype=torch.float64)
        beta=torch.tensor(layer['bn_beta'],dtype=torch.float64)
        mean=torch.tensor(layer['bn_mean'],dtype=torch.float64)
        var=torch.tensor(layer['bn_var'],dtype=torch.float64)
        conv_weight=torch.tensor(layer['conv_weight'],dtype=torch.float64)
        output=composite_layer(input_,gamma,beta,mean,var,conv_weight,eps)
        input_=torch.cat((input_,output),1)
    return input_
def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    """
    Returns torch.Tensor: BN-ReLU-1x1Conv then 2x2 average pool with stride 2 (channels compressed, H and W halved).
    """
    # YOUR CODE HERE
    #batchnorm
    z=batch_norm(x,bn_gamma,bn_beta,bn_mean,bn_var,eps)
    #relu
    y=relu(z)
    #conv1x1
    y1=conv(y,conv_weight)
    #avg_pool
    out=F.avg_pool2d(y1,(2,2),stride=2)
    return out

def densenet_forward(x, weights, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, num_classes) with class logits.
    """
    # YOUR CODE HERE
    x=torch.tensor(x,dtype=torch.float64)
    #stem
    stem_conv=torch.tensor(weights['stem_conv'],dtype=torch.float64)
    x1=conv(x,stem_conv,"3x3")
    n_blocks=len(weights['blocks'])
    n_transitions=len(weights['transitions'])
    y=x1
    j=0
    for i in range(n_blocks):
        #block 1
        y=dense_block(y,weights['blocks'][i],eps)
        #transition layer
        if j<n_transitions:
            g=torch.tensor(weights['transitions'][j]['bn_gamma'],dtype=torch.float64)
            b=torch.tensor(weights['transitions'][j]['bn_beta'],dtype=torch.float64)
            m=torch.tensor(weights['transitions'][j]['bn_mean'],dtype=torch.float64)
            v=torch.tensor(weights['transitions'][j]['bn_var'],dtype=torch.float64)
            conv_weight=torch.tensor(weights['transitions'][j]['conv_weight'],dtype=torch.float64)
            y=transition_layer(y,g,b,m,v,conv_weight,eps)
            j+=1
    # batch norm
    final_bn_var=torch.tensor(weights['final_bn_var'],dtype=torch.float64)
    final_bn_beta=torch.tensor(weights['final_bn_beta'],dtype=torch.float64)
    final_bn_mean=torch.tensor(weights['final_bn_mean'],dtype=torch.float64)
    final_bn_gamma=torch.tensor(weights['final_bn_gamma'],dtype=torch.float64)
    
    y2=batch_norm(y,final_bn_gamma,final_bn_beta,final_bn_mean,final_bn_var,eps)
    y3=relu(y2)
    fc_weight=torch.tensor(weights['fc_weight'],dtype=torch.float64)
    fc_bias=torch.tensor(weights['fc_bias'],dtype=torch.float64)
    n,c,h,w=y3.shape
    #global avg pooling
    y4=y3.mean(dim=(2,3))
    out=y4@fc_weight.T+fc_bias
    return out
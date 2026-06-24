import numpy as np
def relu(x):
    return np.maximum(x,0)
def baseblock(x,W1,W2,Ws=None,downsample=False):
    #block 1 (identity skip, same dims)
    y1=x@W1
    y1=relu(y1)
    y2=y1@W2
    if downsample:
         y=relu(y2+x@Ws)
    else:
        y=relu(y2+x)
    return y
    
def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns: np.ndarray of shape (batch, num_classes) with classification logits
    """
    # YOUR CODE HERE
    #parsing numpy array
    x=np.array(x)
    conv1=np.array(conv1)
    W1_b1=np.array(W1_b1)
    W2_b1=np.array(W2_b1)
    W1_b2=np.array(W1_b2)
    W2_b2=np.array(W2_b2)
    Ws_b2=np.array(Ws_b2)
    fc=np.array(fc)
    #convolution layer
    outconv=x@conv1
    outconv=relu(outconv)
    #stage 1
    y1=baseblock(outconv,W1_b1,W2_b1)
    y1=baseblock(y1,W1_b2,W2_b2,Ws_b2,downsample=True)
    # #stage 2
    # y2=baseblock(y1,W1_b1,W2_b1,Ws_b2,downsample=True)
    # y2=baseblock(y2,W1_b2,W2_b2)
    # #stage 3
    # y3=baseblock(y2,W1_b1,W2_b1,Ws_b2,downsample=True)
    # y3=baseblock(y3,W1_b2,W2_b2)
    # #stage 4 
    # y4=baseblock(y3,W1_b1,W2_b1,Ws_b2,downsample=True)
    # y=baseblock(y4,W1_b2,W2_b2)
    y=y1@fc
    return y
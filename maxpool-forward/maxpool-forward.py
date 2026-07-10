import numpy as np
def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    X=np.array(X)
    h_out=int((X.shape[0]-pool_size)/stride)+1
    w_out=int((X.shape[1]-pool_size)/stride)+1
    output=np.zeros((h_out,w_out))
    for i in range(h_out):
        for j in range(w_out):
            output[i][j]=np.max(X[i*stride:i*stride+pool_size,j*stride:j*stride+pool_size].flatten())
    return output.tolist()
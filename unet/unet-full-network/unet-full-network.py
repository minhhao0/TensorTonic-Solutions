import numpy as np
def encoder_block(H_in,W_in):
    H_out=(H_in-4)//2
    W_out=(W_in-4)//2
    return H_out,W_out
def decoder_block(H_in,W_in):
    H_out=2*H_in-4
    W_out=2*W_in-4
    return H_out,W_out
def bottleneck(H_in,W_in):
    H_out=H_in-4
    W_out=W_in-4
    return H_out,W_out
def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net: trace shape through 4 encoder blocks, bottleneck, 4 decoder blocks, output.
    Each block: two 3x3 unpadded convs (reduce by 4), encoder pools (halve), decoder upsamples (double).
    Returns zero array with correct output shape.
    """
    # Your implementation here
    x=np.array(x)
    B,H,W,C=x.shape
    for i in range(4):
        H,W=encoder_block(H,W)
    H,W=bottleneck(H,W)
    for i in range(4):
        H,W=decoder_block(H,W)
    return np.zeros((B,H,W,num_classes))
    

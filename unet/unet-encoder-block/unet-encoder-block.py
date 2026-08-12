import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    Returns (pool_out, skip_out) as zero arrays with correct shapes.
    """
    # Your implementation here
    x=np.array(x)
    b,h,w,c=x.shape
    pool_tensor=np.zeros((b,(h-4)//2,(w-4)//2,out_channels))
    skip_tensor=np.zeros((b,h-4,w-4,out_channels))
    return pool_tensor,skip_tensor

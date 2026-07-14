import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    # YOUR CODE HERE
    image=np.array(image)
    b,h,w,c=image.shape
    n=(h//patch_size)*(w//patch_size)
    # swap two axis
    patches=image.reshape(b,h//patch_size,patch_size,w//patch_size,patch_size,c).swapaxes(2,3)
    patches=patches.reshape(b,n,patch_size*patch_size*c)
    return patches@W_proj
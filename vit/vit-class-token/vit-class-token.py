import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int, cls_token: np.ndarray = None) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    cls_token: shape (1, 1, D). If None, initialize randomly.
    """
    # YOUR CODE HERE
    if cls_token is None:
        cls_token=np.random.randn(1,1,embed_dim)*0.02
    cls_token=np.array(cls_token)
    patches=np.array(patches)
    result=[]
    for i in range(patches.shape[0]):
        r=np.concatenate((cls_token[0],patches[i]),axis=0)
        result.append(r)
    return np.array(result)
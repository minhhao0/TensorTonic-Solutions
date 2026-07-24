import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    """
    Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """
    # YOUR CODE HERE
    encoder_output=np.array(encoder_output)
    h_cls=encoder_output[:,0,:]
    d=h_cls.shape[-1]
    epsilon=1e-6
    if W_head is None:
        W_head=np.random.randn(d,num_classes)
    mean=np.mean(h_cls,axis=-1,keepdims=True)
    var=np.var(h_cls,axis=-1,keepdims=True)
    std=np.sqrt(var)
    h=(h_cls-mean)/(std+epsilon)
    logits=h@W_head
    return logits
        
import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int, pos_embed: np.ndarray = None) -> np.ndarray:
    """
    Add position embeddings to patch embeddings.
    pos_embed: position embedding of shape (1, N, D). If None, initialize randomly.
    """
    # YOUR CODE HERE
    if pos_embed is None:
        pos_embed=np.random.randn(1,num_patches,embed_dim)*0.02
    patches=np.array(patches)
    pos_embed=np.array(pos_embed)
    pos_embed=pos_embed[0]
    result=[]
    for i in range(patches.shape[0]):
        result.append(patches[i]+pos_embed)
    return np.array(result)
    
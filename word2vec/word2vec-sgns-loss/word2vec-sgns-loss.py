import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the SGNS loss.
    """
    # YOUR CODE HERE
    center_vec=torch.Tensor(center_vec)
    pos_vec=torch.Tensor(pos_vec)
    neg_vecs=torch.Tensor(neg_vecs)
    L=F.softplus(-center_vec@pos_vec)
    for i in range(neg_vecs.shape[0]):
        L+=F.softplus(center_vec@neg_vecs[i])
    return L
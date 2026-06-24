import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    # YOUR CODE HERE
    counts=torch.Tensor(counts)
    N=counts.sum()
    f_w=counts/N
    p_keep=torch.sqrt(t/f_w)
    p_keep=torch.where(p_keep>1,1,p_keep)
    return p_keep
import torch

def noise_distribution(counts: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,), a probability distribution that sums to 1.
    """
    # YOUR CODE HERE
    counts=torch.Tensor(counts)
    numerator=torch.pow(counts,alpha)
    denominator=torch.sum(torch.pow(counts,alpha))
    return numerator/denominator

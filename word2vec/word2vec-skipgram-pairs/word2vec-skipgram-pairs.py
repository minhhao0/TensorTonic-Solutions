import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    # YOUR CODE HERE
    skip_grams=[]
    n=len(token_ids)
    token_ids=torch.Tensor(token_ids)
    for i in range(n):
        center=token_ids[i]
        for j in range(max(0,i-window),min(i+window,n-1)+1):
            if i!=j:
                context=token_ids[j]
                skip_grams.append((center,context))
    if len(skip_grams)==0:
        return torch.zeros((0,2),dtype=torch.int64)
    return torch.Tensor(skip_grams).to(torch.int64)
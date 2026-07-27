import math
import torch

def densenet_channel_counts(stem_channels: int, growth_rate: int, block_layers, compression: float) -> torch.Tensor:
    """
    Returns a 1D int64 torch.Tensor of channel counts at each stage.
    """
    # YOUR CODE HERE
    result=[int(stem_channels)]
    c_trans=0
    for i in range(len(block_layers)):
        c_in=result[-1]+int(block_layers[i]*growth_rate)
        result.append(c_in)
        if i !=len(block_layers)-1:
            #transition 
            c_trans=int(result[-1]*compression)
            result.append(c_trans)
    return torch.tensor(result,dtype=torch.int32)
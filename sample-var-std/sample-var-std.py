import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    N=len(x)
    x=np.array(x)
    mean=np.mean(x)
    var=((x-mean)**2).sum()/(N-1)
    std=np.sqrt(var)
    return (var,std)
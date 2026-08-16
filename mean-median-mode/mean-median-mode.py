import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    counter=Counter(x)
    n=len(x)
    x=np.array(x)
    mean=x.mean()
    median=np.median(x)
    mode = x[0]
    for i in sorted(counter.keys()):
        if i!=mode:
            if counter[i]>counter[mode]:
                mode=i
    return mean,median,mode
        
    
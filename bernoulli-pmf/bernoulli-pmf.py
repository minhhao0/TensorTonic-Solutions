import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    """
    Returns a dictionary with pmf, mean, and variance.
    """
    # Write code here
    x=np.array(x)
    pmf=np.where(x==0,1-p,p)
    mean=p
    var=p*(1-p)
    return {
        "pmf":pmf,
        "mean":float(mean),
        "variance":float(var)
    }
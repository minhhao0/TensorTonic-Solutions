import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Write code here
    pmf=p**k*(1-p)**(n-k)*math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
    cdf=0
    for i in range(k+1):
        cdf+=p**i*(1-p)**(n-i)*math.factorial(n)/(math.factorial(i)*math.factorial(n-i))
    return {
        "pmf":pmf,
        "cdf":cdf
    }
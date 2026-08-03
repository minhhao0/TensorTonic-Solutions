import math
def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    h=0
    N=len(actual_tokens)
    for i in range(N):
        h+=math.log(prob_distributions[i][actual_tokens[i]])
    h=-h/N
    return math.exp(h)
        
        
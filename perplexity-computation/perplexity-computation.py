import math
def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    h=0
    for i in range(len(actual_tokens)):
        h+=math.log(prob_distributions[i][actual_tokens[i]])
    h=-h/len(actual_tokens)
    pp=math.exp(h)
    return pp
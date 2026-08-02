import math
def norm(x):
    s=0
    for i in x:
        s+=i*i
    return math.sqrt(s)
def cosine_sim(x1,x2):
    numerator=0
    for i,j in zip(x1,x2):
        numerator+=i*j
    denominator=norm(x1)*norm(x2)
    return numerator/denominator
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    if label==1:
        return 1-cosine_sim(x1,x2)
    return max(0,cosine_sim(x1,x2)-margin)
    
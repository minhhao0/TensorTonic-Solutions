import math
def clip(p,e):
    if p<e:
        return e
    if p>1-e:
        return 1-e
    return p
def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    # Write code here
    y_pred=[clip(p,eps) for p in y_pred]
    L=[]
    for p_h,p in zip(y_pred,y_true):
        l=-(p*math.log(p_h)+(1-p)*math.log(1-p_h))
        L.append(l)
    return L
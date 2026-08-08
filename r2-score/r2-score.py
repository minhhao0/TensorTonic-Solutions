import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    y_true=np.array(y_true,dtype=np.float64)
    y_pred=np.array(y_pred,dtype=np.float64)
    mean=y_true.mean()
    N=y_true.shape[0]
    if np.all(y_true==np.ones(N)):
        if np.all(y_true==y_pred):
            return 1.0
        else:
            return 0.0
    ss_res=np.sum((y_true-y_pred)*(y_true-y_pred))
    ss_tot=np.sum((y_true-mean)*(y_true-mean))
    return 1-float(ss_res/ss_tot)
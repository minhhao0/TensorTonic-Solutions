import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    # Write code here
    matrix=np.array(matrix,dtype=np.float64)
    amount=np.max(matrix,axis=axis,keepdims=True)
    if norm_type=='l1':
        amount=np.absolute(matrix).sum(axis=axis,keepdims=True)
    if norm_type=='l2':
        amount=np.sqrt(np.sum(matrix**2,axis=axis,keepdims=True))
    return np.divide(matrix,amount,out=np.zeros_like(matrix),where=amount!=0)
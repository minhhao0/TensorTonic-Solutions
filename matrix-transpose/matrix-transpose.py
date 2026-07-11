import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A_T=[]
    rows=len(A)
    columns=len(A[0])
    for j in range(columns):
        row=[]
        for i in range(rows):
            row.append(A[i][j])
        A_T.append(row)
    return  np.array(A_T)
    

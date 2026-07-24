import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    result=[]
    for i in range(seq_len):
        r=[]
        for j in range(d_model):
            if j%2==0:
                r.append(np.sin(i/(base**(j/d_model))))
            else:
                r.append(np.cos(i/(base**((j-1)/d_model))))
        result.append(r)
    return np.array(result)
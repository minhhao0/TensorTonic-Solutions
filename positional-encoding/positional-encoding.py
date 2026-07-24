import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    #Write code here
    result=[]
    embed=np.array([j for j in range(d_model)])
    for i in range(seq_len):
        r=[]
        for j in range(d_model):
            if j%2==0:
                r.append(np.sin(i/(base**(j/d_model))))
            else:
                r.append(np.cos(i/(base**((j-1)/d_model))))
        result.append(r)
    return np.array(result)
    # positions = np.arange(seq_len)[:, np.newaxis]      # shape (seq_len, 1)
    # dims = np.arange(d_model)[np.newaxis, :]            # shape (1, d_model)

    # # exponent uses the "even" index for each pair, i.e. j - (j % 2)
    # angle_rates = 1 / (base ** ((dims - dims % 2) / d_model))
    # angles = positions * angle_rates                    # shape (seq_len, d_model)

    # result = np.zeros((seq_len, d_model))
    # result[:, 0::2] = np.sin(angles[:, 0::2])
    # result[:, 1::2] = np.cos(angles[:, 1::2])

    # return result

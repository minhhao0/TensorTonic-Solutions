import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    result=[]
    for pos in  range(seq_length):
        row=[]
        for i in range(int(d_model/2)):
            exp=2*i/d_model
            numerator=10000**exp
            row.append(np.sin(pos/numerator))
            row.append(np.cos(pos/numerator))
        result.append(row)
    return np.array(result)
                
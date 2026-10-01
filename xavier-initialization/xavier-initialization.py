import math

def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    # Write code here
    L=math.sqrt(6/(fan_in+fan_out))
    result=[]
    for i in range(len(W)):
        row=[]
        for j in range(len(W[i])):
            row.append(W[i][j]*2*L-L)
        result.append(row)
    return result
            
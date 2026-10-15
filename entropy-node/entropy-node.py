import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    cls=set(y)
    h=0
    p=[]
    dic={}
    if y==[]:
        return float(h)
    for c in cls:
        dic[c]=0
        for pred in y:
            if pred==c:
                dic[c]+=1
    for c in cls:
        p.append(dic[c]/len(y))
    for prob in p:
        if prob!=0:
            h-=prob*np.log2(prob)
    return h
        
        
def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    # Write code here
    new=[]
    for i in range(len(values)):
        if values[i] is None:
            right=0
            left=0
            v_left=0
            v_right=0
            j=i-1
            k=i+1
            while j>=0:
                if values[j] is not None:
                    left=j
                    v_left=values[j]
                    break
                else:
                    j-=1
            while k<len(values):
                if values[k] is not None:
                    right=k
                    v_right=values[k]
                    break
                else:
                    k+=1
            new_value=v_left+(i-left)*(v_right-v_left)/(right-left)
            new.append(new_value)
        else:
            new.append(values[i])
    return new
            
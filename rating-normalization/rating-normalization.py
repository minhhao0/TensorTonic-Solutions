def rating_normalization(matrix: list) -> list:
    """
    Returns the mean-centered user-item matrix.
    """
    # Write code here
    users=len(matrix)
    means=[]
    for i in range(users):
        mean=0
        non_zero=0
        for j in range(len(matrix[i])):
            if matrix[i][j]!=0:
                non_zero+=1
                mean+=matrix[i][j]
        if non_zero==0:
            means.append(0)
        else:
            means.append(mean/non_zero)
    for i in range(users):
        mean=means[i]
        for j in range(len(matrix[i])):
            if matrix[i][j]!=0:
                matrix[i][j]-=mean
    return matrix
            
            
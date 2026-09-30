import math

def distance(p1,p2):
    n=0
    for i in range(len(p1)):
        n+=(p1[i]-p2[i])**2
    return math.sqrt(n)
        
def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    # Write code here
    result=[]
    for point in points:
        min=distance(point,centroids[0])
        minidx=0
        for i in range(1,len(centroids)):
            if distance(point,centroids[i])<min:
                min=distance(point,centroids[i])
                minidx=i
        result.append(minidx)
    return result
            
            
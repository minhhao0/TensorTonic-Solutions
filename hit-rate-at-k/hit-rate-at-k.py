def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    hit=0
    total=len(recommendations)
    for i in range(len(recommendations)):
        top_k=set(recommendations[i][:k])
        gt=set(ground_truth[i])
        if gt & top_k:
            hit+=1
            continue
    return hit/total
            
        
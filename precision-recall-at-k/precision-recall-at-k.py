def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k=recommended[:k]
    #calculate precision at k
    numerator=len([i for i in top_k if i in relevant])
    denorminator=k
    presision=numerator/denorminator
    #calculate recall at k
    denorminator=len(relevant)
    recall=numerator/denorminator
    return [presision,recall]
    
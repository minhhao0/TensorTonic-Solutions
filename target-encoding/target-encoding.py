def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    # Write code here
    unique_cat=set(categories)
    cm={}
    count={}
    for cat in unique_cat:
        cm[cat]=0
        count[cat]=0
    for cat in unique_cat:
        for c,v in zip(categories,targets):
            if c==cat:
                cm[c]+=v
                count[c]+=1
    encode=[]
    for cat in categories:
        encode.append(cm[cat]/count[cat])
    return encode
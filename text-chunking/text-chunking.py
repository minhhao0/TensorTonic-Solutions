import math
def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here
    step=chunk_size-overlap
    chunks=[]
    i=0
    if len(tokens)==0:
        return []
    while True:
        chunk=tokens[i*step:i*step+chunk_size]
        chunks.append(chunk)
        i+=1
        if chunk[-1]==tokens[-1]:
            break
    return chunks
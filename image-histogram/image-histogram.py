def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here
    hist=[0]*256
    for i in image:
        for j in i:
            hist[j]+=1
    return hist
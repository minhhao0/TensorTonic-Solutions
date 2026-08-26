def color_to_grayscale(image: list) -> list:
    """
    Returns the luminance value of every RGB pixel.
    """
    # Write code here
    gray_image=[]
    h=len(image)
    for i in range(h):
        r=[]
        w=len(image[i])
        for j in range(w):
            R=image[i][j][0]
            G=image[i][j][1]
            B=image[i][j][2]
            r.append(0.299*R+0.587*G+0.114*B)
        gray_image.append(r)
    return gray_image
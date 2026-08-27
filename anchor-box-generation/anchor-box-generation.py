import math
def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Returns a list of [center_x, center_y, width, height] anchor boxes.
    """
    # Write code here
    #1. Compute the stride
    stride=image_size//feature_size
    #2. Compute the center in image coordinates
    centers=[]
    for i in range(feature_size):
        for j in range(feature_size):
            cx=(j+0.5)*stride
            cy=(i+0.5)*stride
            centers.append((cx,cy))
    #3. compute box width, height
    wh=[]
    for s in scales:
        for r in aspect_ratios:
            w=s*math.sqrt(r)
            h=s/math.sqrt(r)
            wh.append((w,h))
    # Generate anchor box
    anchor_boxes=[]
    for cx,cy in centers:
        for w,h in wh:
            box=[cx-w/2,cy-h/2,cx+w/2,cy+h/2]
            anchor_boxes.append(box)
    return anchor_boxes
            
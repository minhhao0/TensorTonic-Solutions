import math
def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    inter_x1=max(box_a[0],box_b[0])
    inter_y1=max(box_a[1],box_b[1])
    inter_x2=min(box_a[2],box_b[2])
    inter_y2=min(box_a[3],box_b[3])
    inter_height=max(0,inter_y2-inter_y1)
    inter_width=max(0,inter_x2-inter_x1)
    intersection=inter_height*inter_width
    area_a=(box_a[2]-box_a[0])*(box_a[3]-box_a[1])
    area_b=(box_b[2]-box_b[0])*(box_b[3]-box_b[1])
    union=area_a+area_b-intersection
    iou_score=intersection/union
    return iou_score
    
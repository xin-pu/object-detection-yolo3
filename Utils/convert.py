"""
This is a module for convert  rect (box) with different type
"""

import numpy as np


def convert_to_centroid(min_max_boxes):
    """
convert min, max box array to convert_to_centroid box array
    :param min_max_boxes:
    :return:
    """
    min_max_boxes = min_max_boxes.astype(np.float)
    centroid_boxes = np.zeros_like(min_max_boxes)

    x1 = min_max_boxes[:, 0]
    y1 = min_max_boxes[:, 1]
    x2 = min_max_boxes[:, 2]
    y2 = min_max_boxes[:, 3]

    centroid_boxes[:, 0] = (x1 + x2) / 2.0
    centroid_boxes[:, 1] = (y1 + y2) / 2.0
    centroid_boxes[:, 2] = x2 - x1
    centroid_boxes[:, 3] = y2 - y1
    return centroid_boxes


def convert_to_minmax(centroid_boxes):
    """
convert  convert_to_centroid box array to min, max box array
    :param centroid_boxes:
    :return:
    """
    centroid_boxes = centroid_boxes.astype(float)
    minmax_boxes = np.zeros_like(centroid_boxes)

    cx = centroid_boxes[:, 0]
    cy = centroid_boxes[:, 1]
    w = centroid_boxes[:, 2]
    h = centroid_boxes[:, 3]

    minmax_boxes[:, 0] = cx - w / 2.
    minmax_boxes[:, 1] = cy - h / 2.
    minmax_boxes[:, 2] = cx + w / 2.
    minmax_boxes[:, 3] = cy + h / 2.
    return minmax_boxes


def correct_yolo_boxes(boxes, image_h, image_w):
    """
    # Args
        boxes : array, shape of (N, 4)
            [0, 1]-scaled box
    # Returns
        boxes : array shape of (N, 4)
            ([0, image_h], [0, image_w]) - scaled box
    """
    for i in range(len(boxes)):
        boxes[i].x = int(boxes[i].x * image_w)
        boxes[i].w = int(boxes[i].w * image_w)
        boxes[i].y = int(boxes[i].y * image_h)
        boxes[i].h = int(boxes[i].h * image_h)


def convert_to_yolo_boxes(centroid_real_boxes, image_h, image_w):
    """
    Convert
    :param centroid_real_boxes: BoundBoxes
    :param image_h: height of image
    :param image_w: width if image
    """
    centroid_boxes = centroid_real_boxes.astype(float)
    yolo_boxes = np.zeros_like(centroid_boxes)

    cx = centroid_boxes[:, 0]
    cy = centroid_boxes[:, 1]
    w = centroid_boxes[:, 2]
    h = centroid_boxes[:, 3]

    for i in range(len(centroid_boxes)):
        yolo_boxes[:, 0] = cx / image_w
        yolo_boxes[:, 1] = cy / image_h
        yolo_boxes[:, 2] = w / image_w
        yolo_boxes[:, 3] = h / image_h

    return yolo_boxes


def convert_to_real_boxes(centroid_yolo_boxes, image_h, image_w):
    """
    Convert
    :param centroid_yolo_boxes: BoundBoxes
    :param image_h: height of image
    :param image_w: width if image
    """
    centroid_boxes = centroid_yolo_boxes.astype(np.float)
    real_boxes = np.zeros_like(centroid_boxes)

    cx = centroid_boxes[:, 0]
    cy = centroid_boxes[:, 1]
    w = centroid_boxes[:, 2]
    h = centroid_boxes[:, 3]

    for i in range(len(centroid_boxes)):
        real_boxes[:, 0] = cx * image_w
        real_boxes[:, 1] = cy * image_h
        real_boxes[:, 2] = w * image_w
        real_boxes[:, 3] = h * image_h

    return real_boxes


def convert_boxes_to_centroid_boxes(bound_boxes):
    """
    # Args
        boxes : list of BoundBox instances

    # Returns
        centroid_boxes : (N, 4)
        probs : (N,)
    """
    centroid_boxes = []
    probs = []
    for box in bound_boxes:
        centroid_boxes.append([box.x, box.y, box.w, box.h])
        probs.append(box.classes)
    return np.array(centroid_boxes), np.max(np.array(probs), axis=1)


if __name__ == "__main__":
    min_max_box = np.array([[100, 100, 300, 300],
                            [50, 50, 300, 300]])
    centroid_box = convert_to_centroid(min_max_box)
    print(centroid_box)

    minmax_box = convert_to_minmax(centroid_box)
    print(minmax_box)

    yolo_box = convert_to_yolo_boxes(centroid_box, 416, 416)
    print(yolo_box)

    real_box = convert_to_real_boxes(yolo_box, 416, 416)
    print(real_box)

import numpy as np


def create_anchor_array(anchors):
    boxes = []

    n_layer, n_boxes = 3, 3

    for lay in range(n_layer):
        boxes_lay = []
        for i in range(n_boxes):
            boxes_lay.append([anchors[lay * 3 * 2 + 2 * i], anchors[lay * 3 * 2 + 2 * i + 1]])
        boxes.append(boxes_lay)
    return np.array(boxes)


def create_anchor_boxes(anchors):
    """
create anchor_boxes by array
    :param anchors: list of floats
    :return: array, shape of (len(anchors)/2, 4)
            centroid-type
    """
    boxes = []
    n_boxes = int(len(anchors) / 2)
    for i in range(n_boxes):
        boxes.append(np.array([0, 0, anchors[2 * i], anchors[2 * i + 1]]))
    return np.array(boxes)


if __name__ == "__main__":
    test_anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    print(create_anchor_array(test_anchors))
    print(create_anchor_boxes(test_anchors))

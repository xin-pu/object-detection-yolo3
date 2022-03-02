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


def convert_to_encode_box(pattern_shape, input_size, original_min_max_box, match_anchor_box):
    """
convert to box used for prediction
    :param pattern_shape: 52 or 26 or 13
    :param original_min_max_box: min,max box
    :param match_anchor_box: match centroid box
    :param input_size: 416
    :return:
    """
    x1, y1, x2, y2 = original_min_max_box
    _, _, anchor_w, anchor_h = match_anchor_box

    # determine the yolo to be responsible for this bounding box
    rate_w = rate_h = float(pattern_shape) / input_size

    # determine the position of the bounding box on the grid
    x_center = (x1 + x2) / 2.0 * rate_w  # sigma(t_x) + c_x
    y_center = (y1 + y2) / 2.0 * rate_h  # sigma(t_y) + c_y

    # determine the sizes of the bounding box
    w = np.log(max((x2 - x1), 1) / float(anchor_w))  # t_w
    h = np.log(max((y2 - y1), 1) / float(anchor_h))  # t_h

    return [x_center, y_center, w, h]


if __name__ == "__main__":
    test_anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    print(create_anchor_array(test_anchors))
    print(create_anchor_boxes(test_anchors))

    print(convert_to_encode_box(52, 416, [2, 13, 310, 416], [0, 0, 373, 326]))

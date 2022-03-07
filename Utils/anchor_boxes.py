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
b_x = sigma(t_x) + c_x
b_y = sigma(t_y) + c_y
b_w = p_w * e^t_w   => t_w = ln(b_w / p_w)
b_h = p_h * e^t_h   => t_h = ln(b_h / p_h)
    :param pattern_shape: 52 or 26 or 13
    :param original_min_max_box: min,max box
    :param match_anchor_box: match centroid box
    :param input_size: 416
    :return:[tx,ty,tw,th]
    """
    grid_w = grid_h = 1. * input_size / float(pattern_shape)
    b_x1, b_y1, b_x2, b_y2 = original_min_max_box
    _, _, p_w, p_h = match_anchor_box

    # convert to centroid  box
    b_x, b_y = (b_x1 + b_x2) / 2.0, (b_y1 + b_y2) / 2.0
    b_w, b_h = max((b_x2 - b_x1), 1), max((b_y2 - b_y1), 1)
    # determine the yolo to be responsible for this bounding box

    # determine the position of the bounding box on the grid
    x_center = b_x / grid_w   # sigma(t_x) + c_x
    y_center = b_y / grid_h   # sigma(t_y) + c_y

    # determine the sizes of the bounding box
    t_w = np.log(1.0 * b_w / p_w)
    t_h = np.log(1.0 * b_h / p_h)

    return [x_center, y_center, t_w, t_h]


if __name__ == "__main__":
    test_anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    print(create_anchor_array(test_anchors))
    print(create_anchor_boxes(test_anchors))

    print(convert_to_encode_box(52, 416, [2, 13, 310, 416], [0, 0, 373, 326]))

    print(int(np.floor(19.5)))

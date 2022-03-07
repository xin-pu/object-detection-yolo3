"""
This is a module for calculate iou
"""

import tensorflow as tf
from Utils.convert import *


def get_area(min_max_rect):
    return (min_max_rect[2] - min_max_rect[0]) * (
            min_max_rect[3] - min_max_rect[1])


def get_inter_area(min_max_rect_truth, min_max_rect_prediction):
    inter_x_min = max(min_max_rect_truth[0], min_max_rect_prediction[0])
    inter_x_max = min(min_max_rect_truth[2], min_max_rect_prediction[2])

    inter_y_min = max(min_max_rect_truth[1], min_max_rect_prediction[1])
    inter_y_max = min(min_max_rect_truth[3], min_max_rect_prediction[3])

    inter_w = inter_x_max - inter_x_min
    inter_h = inter_y_max - inter_y_min

    rect_int = inter_h * inter_w if inter_w > 0 and inter_h > 0 else 0
    return rect_int


def get_enclosing_area(min_max_rect_truth, min_max_rect_prediction):
    enclosing_x_min = min(min_max_rect_truth[0], min_max_rect_prediction[0])
    enclosing_x_max = max(min_max_rect_truth[2], min_max_rect_prediction[2])

    enclosing_y_min = min(min_max_rect_truth[1], min_max_rect_prediction[1])
    enclosing_y_max = max(min_max_rect_truth[3], min_max_rect_prediction[3])

    area_enclosing = (enclosing_x_max - enclosing_x_min) * (enclosing_y_max - enclosing_y_min)
    return area_enclosing


def get_enclosing_c2(min_max_rect_truth, min_max_rect_prediction):
    enclosing_x_min = min(min_max_rect_truth[0], min_max_rect_prediction[0])
    enclosing_x_max = max(min_max_rect_truth[2], min_max_rect_prediction[2])

    enclosing_y_min = min(min_max_rect_truth[1], min_max_rect_prediction[1])
    enclosing_y_max = max(min_max_rect_truth[3], min_max_rect_prediction[3])

    return (enclosing_x_max - enclosing_x_min) ** 2 + (enclosing_y_max - enclosing_y_min) ** 2


# IOU Loss将4个点构成的box看成一个整体做回归
def get_iou(min_max_rect_truth, min_max_rect_prediction):
    # get area of predict bounding box
    area_prediction = get_area(min_max_rect_prediction)
    # get area of truth bounding box
    area_truth = get_area(min_max_rect_truth)

    # get int area
    area_inter = get_inter_area(min_max_rect_truth, min_max_rect_prediction)

    # get union area
    area_union = area_prediction + area_truth - area_inter

    return area_inter / area_union


def get_tf_iou(boxes_1, boxes_2):
    """
    calculate regression loss using iou
    :param boxes_1: boxes_1 shape is [x, y, w, h]
    :param boxes_2: boxes_2 shape is [x, y, w, h]
    :return:
    """
    # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
    boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                         boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
    boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                         boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
    boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                         tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
    boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                         tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

    # calculate area of boxes_1 boxes_2
    boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
    boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

    # calculate the two corners of the intersection
    left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
    right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

    # calculate area of intersection
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    # calculate union area
    union_area = boxes_1_area + boxes_2_area - inter_area

    # calculate iou add epsilon in denominator to avoid dividing by 0
    iou = inter_area / (union_area + tf.keras.backend.epsilon())

    return iou


# 在IOU基础上优化两个框不想交的情况
def get_giou(min_max_rect_truth, min_max_rect_prediction):
    # get area of predict bounding box
    area_prediction = get_area(min_max_rect_prediction)
    # get area of truth bounding box
    area_truth = get_area(min_max_rect_truth)

    # get int area
    area_int = get_inter_area(min_max_rect_truth, min_max_rect_prediction)

    # get enclosing area
    area_enclosing = get_enclosing_area(min_max_rect_truth, min_max_rect_prediction)

    # get union area
    area_union = area_prediction + area_truth - area_int

    iou = area_int / area_union

    return iou - (area_enclosing - area_union) / area_enclosing


def get_tf_giou(boxes_1, boxes_2):
    """
    calculate regression loss using giou
    :param boxes_1: boxes_1 shape is [x, y, w, h]
    :param boxes_2: boxes_2 shape is [x, y, w, h]
    :return:
    """
    # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
    boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                         boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
    boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                         boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
    boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                         tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
    boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                         tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

    # calculate area of boxes_1 boxes_2
    boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
    boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

    # calculate the two corners of the intersection
    left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
    right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

    # calculate area of intersection
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    # calculate union area
    union_area = boxes_1_area + boxes_2_area - inter_area

    # calculate iou add epsilon in denominator to avoid dividing by 0
    iou = inter_area / (union_area + tf.keras.backend.epsilon())

    # calculate the upper left and lower right corners of the minimum closed convex surface
    enclose_left_up = tf.minimum(boxes_1[..., :2], boxes_2[..., :2])
    enclose_right_down = tf.maximum(boxes_1[..., 2:], boxes_2[..., 2:])

    # calculate width and height of the minimun closed convex surface
    enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # calculate area of the minimun closed convex surface
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

    # calculate the giou add epsilon in denominator to avoid dividing by 0
    giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + tf.keras.backend.epsilon())

    return giou


# 引入最小外接框来最大化重叠面积的惩罚项修改成最小化两个BBox中心点的标准化距离从而加速损失的收敛过程
def get_diou(min_max_rect_truth, min_max_rect_prediction):
    center_x_truth, center_y_truth = (min_max_rect_truth[2] - min_max_rect_truth[0]) / 2, (
            min_max_rect_truth[3] - min_max_rect_truth[1]) / 2

    center_x_prediction, center_y_prediction = (min_max_rect_prediction[2] - min_max_rect_prediction[0]) / 2, (
            min_max_rect_prediction[3] - min_max_rect_prediction[1]) / 2

    p2 = (center_x_truth - center_x_prediction) ** 2 + (center_y_truth - center_y_prediction) ** 2
    c2 = get_enclosing_c2(min_max_rect_truth, min_max_rect_prediction)
    return get_iou(min_max_rect_truth, min_max_rect_prediction) - float(p2) / c2


def get_tf_diou(boxes_1, boxes_2):
    """
    calculate regression loss using diou
    :param boxes_1: boxes_1 shape is [x, y, w, h]
    :param boxes_2: boxes_2 shape is [x, y, w, h]
    :return:
    """
    # calculate center distance
    center_distance = tf.reduce_sum(tf.square(boxes_1[..., :2] - boxes_2[..., :2]), axis=-1)

    # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
    boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                         boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
    boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                         boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
    boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                         tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
    boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                         tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

    # calculate area of boxes_1 boxes_2
    boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
    boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

    # calculate the two corners of the intersection
    left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
    right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

    # calculate area of intersection
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    # calculate union area
    union_area = boxes_1_area + boxes_2_area - inter_area

    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = inter_area / (union_area + tf.keras.backend.epsilon())

    # calculate the upper left and lower right corners of the minimum closed convex surface
    enclose_left_up = tf.minimum(boxes_1[..., :2], boxes_2[..., :2])
    enclose_right_down = tf.maximum(boxes_1[..., 2:], boxes_2[..., 2:])

    # calculate width and height of the minimun closed convex surface
    enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # calculate enclosed diagonal distance
    enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)

    # calculate diou add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * center_distance / (enclose_diagonal + tf.keras.backend.epsilon())

    return diou


# 在DIOU的基础上将Bounding box的纵横比考虑进损失函数中，进一步提升了回归精度
def get_ciou(min_max_rect_truth, min_max_rect_prediction):
    # Todo
    pass


def get_tf_ciou(boxes_1, boxes_2):
    # Todo
    pass


def get_eiou(min_max_rect_truth, min_max_rect_prediction):
    # Todo
    pass


def get_tf_eiou(boxes_1, boxes_2):
    # Todo
    pass


if __name__ == "__main__":
    minmax_rect_arr_a = np.array([[50., 50., 200., 200.]])
    minmax_rect_arr_b = np.array([[80., 160., 220., 220.]])

    test_inter_area = get_inter_area(minmax_rect_arr_a[0], minmax_rect_arr_b[0])
    print(test_inter_area)

    enclosing_area = get_enclosing_area(minmax_rect_arr_a[0], minmax_rect_arr_b[0])
    print(enclosing_area)

    iou_ = get_iou(minmax_rect_arr_a[0], minmax_rect_arr_b[0])
    print("{:.2f}".format(iou_))

    giou_ = get_giou(minmax_rect_arr_a[0], minmax_rect_arr_b[0])
    print("{:.2f}".format(giou_))

    diou_ = get_diou(minmax_rect_arr_a[0], minmax_rect_arr_b[0])
    print("{:.2f}".format(diou_))

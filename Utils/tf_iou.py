import tensorflow as tf


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


def get_tf_ciou( boxes_1, boxes_2):
    """
    calculate regression loss using ciou
    :param boxes_1: boxes_1 shape is [x, y, w, h]
    :param boxes_2: boxes_2 shape is [x, y, w, h]
    :return:
    """
    # calculate center distance
    center_distance = tf.reduce_sum(tf.square(boxes_1[..., :2] - boxes_2[..., :2]), axis=-1)

    v = 4 * tf.square(
        tf.math.atan2(boxes_1[..., 2], boxes_1[..., 3]) - tf.math.atan2(boxes_2[..., 2], boxes_2[..., 3])) / (
                    math.pi * math.pi)

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

    # calculate diou
    diou = iou - 1.0 * center_distance / (enclose_diagonal + tf.keras.backend.epsilon())

    # calculate param v and alpha to CIoU
    alpha = v / (1.0 - iou + v)

    # calculate ciou
    ciou = diou - alpha * v

    return ciou

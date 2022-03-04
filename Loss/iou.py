"""
This is a module for calculate iou
"""

from Utils.convert import *


def get_iou_from_minmax_rect(min_max_rect_a, min_max_rect_b):
    """
get iou
    :param min_max_rect_a: 1-dimensional np array with min max rect
    :param min_max_rect_b: 1-dimensional np array with min max rect
    :return: iou
    """
    x_a = max(min_max_rect_a[0], min_max_rect_b[0])
    y_a = max(min_max_rect_a[1], min_max_rect_b[1])
    x_b = min(min_max_rect_a[2], min_max_rect_b[2])
    y_b = min(min_max_rect_a[3], min_max_rect_b[3])

    # 计算交集部分面积
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # 计算预测值和真实值的面积
    rec_a_area = (min_max_rect_a[2] - min_max_rect_a[0] + 1) * (min_max_rect_a[3] - min_max_rect_a[1] + 1)
    rec_b_area = (min_max_rect_b[2] - min_max_rect_b[0] + 1) * (min_max_rect_b[3] - min_max_rect_b[1] + 1)

    # 计算IOU
    iou = inter_area / float(rec_a_area + rec_b_area - inter_area)
    return iou


def get_iou_from_centroid_rect(centroid_rect_a, centroid_rect_b):
    """
get iou
    :param centroid_rect_a: 1-dimensional np array with centroid rect
    :param centroid_rect_b: 1-dimensional np array with centroid rect
    :return: iou
    """

    minmax_a = convert_to_minmax(centroid_rect_a.reshape(-1, 4))[0]
    minmax_b = convert_to_minmax(centroid_rect_b.reshape(-1, 4))[0]
    return get_iou_from_minmax_rect(minmax_a, minmax_b)


def get_iou_from_centroid_rect_2(box1, box2):
    """

    :param box1:
    :param box2:
    :return:
    """

    def _interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    _, _, w1, h1 = box1.reshape(-1, )
    _, _, w2, h2 = box2.reshape(-1, )
    x1_min, y1_min, x1_max, y1_max = convert_to_minmax(box1.reshape(-1, 4)).reshape(-1, )
    x2_min, y2_min, x2_max, y2_max = convert_to_minmax(box2.reshape(-1, 4)).reshape(-1, )

    intersect_w = _interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = _interval_overlap([y1_min, y1_max], [y2_min, y2_max])
    intersect = intersect_w * intersect_h
    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


if __name__ == "__main__":
    minmax_rect_arr_a = np.array([[50, 50, 300, 300]])
    minmax_rect_arr_b = np.array([[60, 60, 320, 320]])

    centroid_rect_arr_a = convert_to_centroid(minmax_rect_arr_a)
    centroid_rect_arr_b = convert_to_centroid(minmax_rect_arr_b)

    iou_by_min_max_rect = get_iou_from_minmax_rect(minmax_rect_arr_a[0], minmax_rect_arr_b[0])
    iou_by_centroid_rect = get_iou_from_centroid_rect(centroid_rect_arr_a[0], centroid_rect_arr_b[0])
    print("iou:\t{:.4f}\r\niou:\t{:.4f}".format(iou_by_min_max_rect, iou_by_centroid_rect))

    iou_2 = get_iou_from_centroid_rect_2(centroid_rect_arr_a[0], centroid_rect_arr_b[0])
    print("iou:\t{:.4f}".format(iou_2))

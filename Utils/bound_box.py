from iou import *


class BoundBox:

    def __init__(self, x, y, w, h, classes=None):
        """
Initial Bound Box with centroid rect: x, y, w ,h
        :param x:
        :param y:
        :param w:
        :param h:
        :param classes:
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.classes = classes

    def get_label(self):
        return np.argmax(self.classes)

    def get_score(self):
        return self.classes[self.get_label()]

    def get_iou_with_bound_box(self, bound_box):
        b1 = self.as_centroid_rect()
        b2 = bound_box.as_centroid_rect()
        return get_iou_from_centroid_rect(b1, b2)

    def get_match_bound_box(self, bound_boxes):
        match_index = -1
        max_iou = -1
        for i, box in enumerate(bound_boxes):
            temp_iou = self.get_iou_with_bound_box(box)
            if max_iou < temp_iou:
                match_index = i
                max_iou = temp_iou
        return bound_boxes[match_index]

    def as_centroid_rect(self):
        return np.array([self.x, self.y, self.w, self.h])

    def as_min_max_rect(self):
        centroid_rect = self.as_centroid_rect()
        return convert_to_minmax(centroid_rect.reshape(-1, 4))[0]

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        info += "centroid_rect:\t{}\r\n".format(self.as_centroid_rect())
        info += "min_max_rect:\t{}\r\n".format(self.as_min_max_rect())
        info += "class:\t{}\r\n".format(self.get_label())
        info += "score:\t{}\r\n".format(self.get_score())
        return info


if __name__ == '__main__':
    test_box = BoundBox(100, 200, 30, 40, [0, 0, 0, 0.8, 0.7, 0])
    test_box2 = BoundBox(100, 200, 25, 40, [0, 0, 0, 0.8, 0.7, 0])
    print(test_box)

    res = test_box.get_match_bound_box([test_box2])
    print(res)

    iou = test_box.get_iou_with_bound_box(test_box2)
    print(iou)

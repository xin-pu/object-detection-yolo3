from Utils.iou import *


class BoundBox:

    def __init__(self, x, y, w, h, object_ness=None, classes=None):
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
        self.object_ness = object_ness
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
        return match_index, bound_boxes[match_index]

    def get_match_anchor_box(self, anchor_boxes):
        shift_box = BoundBox(0, 0, self.w, self.h)
        bound_boxes = [BoundBox(0, 0, box[2], box[3]) for box in anchor_boxes]
        match_index, bound = shift_box.get_match_bound_box(bound_boxes)
        return match_index, anchor_boxes[match_index]

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


def nms_boxes(boxes, nms_threshold=0.3, obj_threshold=0.3):
    """
    # Args
        boxes : list of BoundBox

    # Returns
        boxes : list of BoundBox
            non maximum supressed BoundBox instances
    """
    if len(boxes) == 0:
        return boxes
    # suppress non-maximal boxes
    n_classes = len(boxes[0].classes)
    for c in range(n_classes):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if boxes[index_i].get_iou_with_bound_box(boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    return boxes


if __name__ == '__main__':
    test_box = BoundBox(100, 200, 30, 40, [0, 0, 0, 0.8, 0.7, 0])
    test_box2 = BoundBox(100, 200, 25, 40, [0, 0, 0, 0.8, 0.7, 0])
    test_box3 = BoundBox(100, 200, 30, 41, [0, 0, 0, 0.8, 0.7, 0])
    print(test_box)

    index, res = test_box.get_match_bound_box([test_box2, test_box3])
    print(res)

    _, anchor_box = test_box.get_match_anchor_box([[0, 0, 25, 40], [0, 0, 30, 41]])
    print(anchor_box)

    iou = test_box.get_iou_with_bound_box(test_box2)
    print(iou)

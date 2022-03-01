
from Utils.box import *
import numpy as np
import cv2

IDX_X = 0
IDX_Y = 1
IDX_W = 2
IDX_H = 3
IDX_OBJECTNESS = 4
IDX_CLASS_PROB = 5


# 输出预测
class YoloDetector(object):

    def __init__(self, model, anchors, image_size=416, obj_thresh=0.5, nms_thresh=0.5):
        self.yolo_net = model
        self.anchors = anchors
        self.image_size = image_size
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh

    def detect_from_file(self, image_source, cls_threshold=0.5):
        image = cv2.imread(image_source)
        image = np.array(image)
        return self.detect(image, cls_threshold)

    def detect(self, image, cls_threshold=0.5):
        """
        # Args
            image : array, shape of (H, W, 3)
            anchors : list, length of 18
            net_size : int
        # Returns
            boxes : array, shape of (N, 4)
                (x1, y1, x2, y2) ordered boxes
            labels : array, shape of (N,)
            probs : array, shape of (N,)
        """

        image_h, image_w, _ = image.shape
        # 1. Pre process [1,416,416,3]
        new_image = self.pre_process(image)

        # 2. Predict [52*52,26*26,13*13]
        yolo_res = self.yolo_net.predict(new_image)

        # 3.Resolve and Get Boxes
        all_boxes = self.post_process(yolo_res, image_h, image_w)
        print(len(all_boxes))
        if len(all_boxes) > 0:
            boxes, probs = boxes_to_array(all_boxes)
            boxes = convert_to_minmax(boxes)
            labels = np.array([b.get_label() for b in all_boxes])

            boxes = boxes[probs >= cls_threshold]
            labels = labels[probs >= cls_threshold]
            probs = probs[probs >= cls_threshold]
        else:
            boxes, labels, probs = [], [], []
        return boxes, labels, probs

    # Convert image to input shape, and normalizing
    def pre_process(self, image):
        preprocess_img = cv2.resize(image / 255., (self.image_size, self.image_size))
        return np.expand_dims(preprocess_img, axis=0)

    def post_process(self, yolo_res, original_height, original_width):
        boxes = []
        # 小尺度  0， 中尺度 1， 大尺度 2，对应anchor 从小到大
        lay = 0
        for yolo_scale in yolo_res:
            boxes += self.decode(yolo_scale[0], self.anchors[lay])
            lay += 1

        # 2. correct box-scale to image size
        correct_yolo_boxes(boxes, original_height, original_width)

        # 3. suppress non-maximal boxes
        nms_boxes(boxes, self.nms_thresh, self.obj_thresh)
        return boxes

    # 从单尺度 Y_Pre 中解码 并抑制 得到Boxes
    def decode(self, y_pre_one_scale, anchors, nb_box=3):
        """
        # Args
            netout : (n_rows, n_cols, 3, 4+1+n_classes)
            anchors

        """
        n_rows, n_cols = y_pre_one_scale.shape[:2]
        y_pre_one_scale = y_pre_one_scale.reshape((n_rows, n_cols, nb_box, -1))
        boxes = []
        for row in range(n_rows):
            for col in range(n_cols):
                for b in range(nb_box):
                    # 1. decode

                    x, y, w, h = self.decode_coordinate(y_pre_one_scale, row, col, b, anchors[b])
                    objectness, classes = self.activate_probs(y_pre_one_scale[row, col, b, IDX_OBJECTNESS],
                                                              y_pre_one_scale[row, col, b, IDX_CLASS_PROB:],
                                                              self.obj_thresh)

                    # 2. scale normalize
                    x /= n_cols
                    y /= n_rows
                    w /= self.image_size
                    h /= self.image_size

                    if objectness > self.obj_thresh:
                        box = BoundBox(x, y, w, h, objectness, classes)
                        boxes.append(box)

        return boxes

    @staticmethod
    def decode_coordinate(netout, row, col, b, anchors):
        x, y, w, h = netout[row, col, b, :IDX_H + 1]

        x = col + sigmoid(x)
        y = row + sigmoid(y)
        w = anchors[0] * np.exp(w)
        h = anchors[1] * np.exp(h)

        return x, y, w, h

    @staticmethod
    def activate_probs(objectness, classes, obj_thresh=0.3):
        """
        # Args
            objectness : scalar
            classes : (n_classes, )

        # Returns
            objectness_prob : (n_rows, n_cols, n_box)
            classes_conditional_probs : (n_rows, n_cols, n_box, n_classes)
        """
        # 1. sigmoid activation
        objectness_prob = sigmoid(objectness)
        classes_probs = sigmoid(classes)
        # 2. conditional probability
        classes_conditional_probs = classes_probs * objectness_prob
        # 3. thresholding
        classes_conditional_probs *= objectness_prob > obj_thresh
        return objectness_prob, classes_conditional_probs


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


if __name__ == "__main__":
    pass
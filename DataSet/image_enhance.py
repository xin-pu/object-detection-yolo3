# -*- coding: utf-8 -*-

import cv2
import numpy as np

from DataSet.pascalvoc_parser import PascalVocParser

np.random.seed(1337)


class ImageEnhance(object):
    def __init__(self, w, h, enhance):
        """
        :param w: int
        :param h: int
        :param enhance: bool
        """
        self.w = w
        self.h = h
        self.enhance = enhance

    def get_image(self, image_filename, boxes):
        """
        get image from filename
        :param image_filename: str
        :param boxes:  array, shape of (N, 4) - min max box
        :return:
        image : 3d-array, shape of (h, w, 3)
        boxes_ : array, same shape of boxes
                enhance & resized bounding box
        """
        # 1. read image file
        image = cv2.imread(image_filename)

        # 2. make enhance on image
        boxes = np.copy(boxes)
        if self.enhance:
            image, boxes = self.make_enhance_on_image(image, boxes)

        # 3. resize image
        image, boxes = self.resize_image(image, boxes, self.w, self.h)
        return image, boxes

    @staticmethod
    def make_enhance_on_image(image, boxes):
        h, w, _ = image.shape

        # 图像随机放大  范围 [1~1.1]
        scale = np.random.uniform() / 10. + 1.  # [1~1.1]
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        # 图像 放大后增加的像素
        max_off_x = (scale - 1.) * w
        max_off_y = (scale - 1.) * h

        off_x = int(np.random.uniform() * max_off_x)
        off_y = int(np.random.uniform() * max_off_y)

        image = image[off_y: (off_y + h), off_x: (off_x + w)]

        # flip the image
        flip = np.random.binomial(1, .5)
        is_flip = True if flip > 0.5 else False
        if flip:
            image = cv2.flip(image, 1)  # 水平翻转

        # fix object's position and size
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1 * scale - off_x)
            y1 = int(y1 * scale - off_x)

            x2 = int(x2 * scale - off_x)
            y2 = int(y2 * scale - off_y)

            if is_flip:
                xmin = x1
                x1 = w - x2
                x2 = w - xmin
            new_boxes.append([x1, y1, x2, y2])
        return image, np.array(new_boxes)

    @staticmethod
    def resize_image(image, boxes, desired_w, desired_h):
        h, w, _ = image.shape

        # resize the image to standard size
        image = cv2.resize(image, (desired_h, desired_w))
        # image = image[:, :, ::-1]

        # fix object's position and size
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = int(x1 * float(desired_w) / w)
            x1 = max(min(x1, desired_w), 0)
            x2 = int(x2 * float(desired_w) / w)
            x2 = max(min(x2, desired_w), 0)

            y1 = int(y1 * float(desired_h) / h)
            y1 = max(min(y1, desired_h), 0)
            y2 = int(y2 * float(desired_h) / h)
            y2 = max(min(y2, desired_h), 0)

            new_boxes.append([x1, y1, x2, y2])
        return image, np.array(new_boxes)


if __name__ == '__main__':
    dataset_dire = r"F:\PASCALVOC\VOC2007\JPEGImages"
    ann_filename = r"F:\PASCALVOC\VOC2007\Annotations\000023.xml"
    class_labels = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                    "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person",
                    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    parser = PascalVocParser(dataset_dire, class_labels)
    ann = parser.get_annotation(ann_filename)
    print(ann)
    res_image, res_boxes = ImageEnhance(416, 416, True).get_image(ann.image_filename, ann.boxes)
    print(res_boxes)
    cv2.imshow("hello", res_image)
    cv2.waitKey(2000)

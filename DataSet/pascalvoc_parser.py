# -*- coding: utf-8 -*-

import os
import numpy as np
from xml.etree.ElementTree import parse


class Annotation(object):
    image_filename = None
    image_size = None
    boxes = []
    labels_str = []
    labels_code = []

    def __init__(self):
        pass

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            info += "{}:\t{}\r\n".format(key, value)
        return info


class PascalVocParser(object):

    def __init__(self, root_dire, class_labels):
        self.root_dire = root_dire
        self.class_labels = class_labels
        self.images_dire = root_dire

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            info += "{}:\t{}\r\n".format(key, value)
        return info

    def get_annotation(self, annotation_file):

        tree, root = self.get_tree_root(annotation_file)

        annotation = Annotation()

        annotation.image_filename = os.path.join(self.images_dire, PascalVocParser.get_image_filename(root))
        annotation.image_size = PascalVocParser.get_image_size(tree)
        annotation.labels_str, annotation.labels_code, annotation.boxes = \
            PascalVocParser.get_objects(root, self.class_labels)

        return annotation

    @staticmethod
    def get_tree_root(filename):
        tree = parse(filename)
        root = tree.getroot()
        return tree, root

    @staticmethod
    def get_image_filename(root):
        return root.find("filename").text

    @staticmethod
    def get_image_size(tree):
        height, width = None, None
        for elem in tree.iter():
            if 'width' in elem.tag:
                width = int(elem.text)
            if 'height' in elem.tag:
                height = int(elem.text)
        return [width, height]

    @staticmethod
    def get_objects(root, classes):
        obj_tags = root.findall("object")
        label_list = []
        code_label_list = []
        boxes = None
        for t in obj_tags:
            label = t.find("name").text
            name_index = classes.index(label)
            box_tag = t.find("bndbox")
            x1 = float(box_tag.find("xmin").text)
            y1 = float(box_tag.find("ymin").text)
            x2 = float(box_tag.find("xmax").text)
            y2 = float(box_tag.find("ymax").text)
            label_list.append(label)
            code_label_list.append(name_index)
            if boxes is None:
                boxes = np.array([x1, y1, x2, y2]).reshape(-1, 4)
            else:
                box = np.array([x1, y1, x2, y2]).reshape(-1, 4)
                boxes = np.concatenate([boxes, box])
        return label_list, code_label_list, boxes


if __name__ == '__main__':
    dataset_dire = r"F:\PASCALVOC\VOC2007\JPEGImages"
    ann_filename = r"F:\PASCALVOC\VOC2007\Annotations\000042.xml"
    pascal_voc_2007_labels = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                              "bus", "car", "cat", "chair", "cow",
                              "diningtable", "dog", "horse", "motorbike", "person",
                              "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    pascal_voc_parser = PascalVocParser(dataset_dire, pascal_voc_2007_labels)
    print(pascal_voc_parser)
    ann = pascal_voc_parser.get_annotation(ann_filename)
    print(ann)

# -*- coding: utf-8 -*-

import os
import numpy as np
from xml.etree.ElementTree import parse


class Annotation(object):
    image_filename = None
    image_size = None
    boxes = []
    labels = []
    code_labels = []

    def __init__(self):
        pass

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        info += "filename:\t{}\r\n".format(self.image_filename)
        info += "size:\t({})\r\n".format(self.image_size)
        info += "labels:\t({})\r\n".format(self.labels)
        info += "labels:\t({})\r\n".format(self.code_labels)
        info += "boxes:\t({})\r\n".format(self.boxes)
        return info


class PascalVocParser(object):
    pascal_voc_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                          "bus", "car", "cat", "chair", "cow",
                          "diningtable", "dog", "horse", "motorbike", "person",
                          "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def __init__(self, root_dire):
        self.root_dire = root_dire
        self.annotations_dire = r"{}\Annotations".format(root_dire)
        self.images_dire = r"{}\JPEGImages".format(root_dire)

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        info += "root dire:\t{}\r\n".format(self.root_dire)
        info += "classes:\t{}\r\n".format(self.pascal_voc_classes)
        return info

    def get_annotation(self, annotation_file):
        """

        :rtype: Annotation
        """

        tree, root = self.get_tree_root(annotation_file)

        annotation = Annotation()

        annotation.image_filename = os.path.join(self.images_dire, PascalVocParser.get_image_filename(root))
        annotation.image_size = PascalVocParser.get_image_size(tree)
        annotation.labels, annotation.code_labels, annotation.boxes = \
            PascalVocParser.get_objects(root, pascal_voc_parser.pascal_voc_classes)

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
    dataset_dire = r"F:\PASCALVOC\VOC2007"
    ann_filename = r"F:\PASCALVOC\VOC2007\Annotations\000042.xml"
    pascal_voc_parser = PascalVocParser(dataset_dire)
    print(pascal_voc_parser)
    ann = pascal_voc_parser.get_annotation(ann_filename)
    print(ann)

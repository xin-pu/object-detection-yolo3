import glob
import os
import numpy as np

from DataSet.pascalvoc_parser import PascalVocParser


def get_annotation(ann_filename, image_dire, labels):
    parser = PascalVocParser(labels, image_dire)
    annotation = parser.get_annotation(ann_filename)
    return annotation


class BatchGenerator(object):
    def __init__(self,
                 model_configs,
                 train_configs,
                 train_batch):
        self.label_names = model_configs.labels
        self.classes = len(self.label_names)
        self.input_size = model_configs.input_size
        self.anchors = model_configs.anchors
        self.anchors_boxes = model_configs.anchor_boxes
        self.anchors_array = model_configs.anchor_array
        self.pattern_shape = model_configs.pattern_shape

        annot_folder = train_configs.train_annot_folder if train_batch else train_configs.valid_annot_folder
        image_folder = train_configs.train_image_folder if train_batch else train_configs.valid_image_folder
        annot_filenames = self.get_ann_filenames(annot_folder) if train_batch \
            else self.get_ann_filenames(annot_folder)[0:train_configs.valid_size]
        self.annot_filenames = annot_filenames
        self.img_dir = image_folder

        self.enhance = train_configs.enhance
        self.shuffle = True

        self.batch_size = batch_size = train_configs.batch_size

        self.random_net_size = train_configs.random_net_size
        self.save_folder = train_configs.save_folder
        self.steps_per_epoch = int(len(self.annot_filenames) / batch_size)
        self.data_length = len(self.annot_filenames)

        self._epoch = 0

    @staticmethod
    def get_ann_filenames(folder):
        ann_filenames = glob.glob(os.path.join(folder, "*.xml"))
        return ann_filenames

    def next_batch(self):
        classes = len(self.annot_filenames)
        i = 0
        while True:

            inputs = np.zeros((self.batch_size, self.input_size, self.input_size, 3))
            # [52*52 26*26 13*13]
            y_trues = create_empty_y_true(self.batch_size, self.pattern_shape, self.classes, 3)

            for batch_index in range(self.batch_size):
                # 随机打乱标签文件名列表
                if i == 0:
                    np.random.shuffle(self.annot_filenames)

                annotation = get_annotation(self.annot_filenames[i], self.img_dir, self.label_names)
                image_file, boxes, coded_labels = annotation.filename, annotation.boxes, annotation.coded_labels
                # boxes: x1,y1,x2,y2 original

                img, boxes = self.get_image(image_file, boxes)
                # boxes: x1,y1,x2,y2 resize as image resize 416
                inputs[batch_index, ...] = img

                # 更新 y_trues
                for original_box, label in zip(boxes, coded_labels):
                    max_anchor, scale_index, box_index = find_match_anchor(original_box, self.anchors_boxes)

                    coded_box = encode_box(self.pattern_shape[scale_index], original_box, max_anchor,
                                           self.input_size)
                    assign_box(y_trues, scale_index, box_index, coded_box, label)

                i = (i + 1) % classes

            # y_trues特征图尺寸依次为小 中 大
            # [52*52 26*26 13*13]
            outputs = [y_trues[0], y_trues[1], y_trues[2]]
            yield inputs, outputs

    def get_next_batch(self):
        classes = len(self.annot_filenames)
        i = 0
        while True:

            inputs = np.zeros((self.batch_size, self.input_size, self.input_size, 3))
            y_trues = create_empty_y_true(self.batch_size, self.pattern_shape, self.classes, 3)

            for batch_index in range(self.batch_size):
                # 随机打乱标签文件名列表
                if i == 0:
                    np.random.shuffle(self.annot_filenames)
                annotation = get_annotation(self.annot_filenames[i], self.img_dir, self.label_names)
                image_file, boxes, coded_labels = annotation.filename, annotation.boxes, annotation.coded_labels

                img, boxes = self.get_image(image_file, boxes)
                inputs[batch_index, ...] = img

                # 更新 y_trues
                for original_box, label in zip(boxes, coded_labels):
                    max_anchor, scale_index, box_index = find_match_anchor(original_box, self.anchors_boxes)
                    coded_box = encode_box(self.pattern_shape[scale_index], original_box, max_anchor,
                                           self.input_size)
                    assign_box(y_trues, scale_index, box_index, coded_box, label)

                i = (i + 1) % classes

            # y_trues特征图尺寸依次为大、中、小
            # [13*13 26*26 52*52]
            outputs = [y_trues[0], y_trues[1], y_trues[2]]
            return inputs, outputs

    def get_image(self, image_file, boxes):
        img_augmenter = ImageAugment(self.input_size, self.input_size, self.enhance)
        img, boxes = img_augmenter.get_image(image_file, boxes)
        return normalize(img), boxes


def find_match_anchor(box, anchor_boxes):
    """
    # Args
        box : array, shape of (4,)
        anchor_boxes : array, shape of (9, 4)
    """
    from utils.box import find_match_box
    x1, y1, x2, y2 = box
    shifted_box = np.array([0, 0, x2 - x1, y2 - y1])

    max_index = find_match_box(shifted_box, anchor_boxes)
    max_anchor = anchor_boxes[max_index]

    lay_index = max_index // 3
    box_index = max_index % 3
    return max_anchor, lay_index, box_index


def create_empty_y_true(batch_size, pattern_shape, n_classes, n_boxes=3):
    y_trues = [np.zeros((batch_size, pattern_shape[i], pattern_shape[i],
                         n_boxes, 5 + n_classes)).astype(np.float32) for i in range(n_boxes)]

    return y_trues


def encode_box(pattern_shape, original_box, anchor_box, input_size):
    x1, y1, x2, y2 = original_box
    _, _, anchor_w, anchor_h = anchor_box

    # determine the yolo to be responsible for this bounding box
    rate_w = rate_h = float(pattern_shape) / input_size

    # determine the position of the bounding box on the grid
    x_center = (x1 + x2) / 2.0 * rate_w  # sigma(t_x) + c_x
    y_center = (y1 + y2) / 2.0 * rate_h  # sigma(t_y) + c_y

    # determine the sizes of the bounding box
    w = np.log(max((x2 - x1), 1) / float(anchor_w))  # t_w
    h = np.log(max((y2 - y1), 1) / float(anchor_h))  # t_h
    # print("x1, y1, x2, y2", x1, y1, x2, y2)
    # print("xc, yc, w, h", x_center, y_center, w, h)

    return [x_center, y_center, w, h]


def assign_box(yolo_true_output, lay_index, box_index, box, label):
    center_x, center_y, _, _ = box

    # determine the location of the cell responsible for this object
    grid_x = int(np.floor(center_x))
    grid_y = int(np.floor(center_y))

    # assign ground truth x, y, w, h, confidence and class probability to y_batch
    yolo_true_output[lay_index][:, grid_y, grid_x, box_index, 0:4] = box
    yolo_true_output[lay_index][:, grid_y, grid_x, box_index, 4] = 1.
    yolo_true_output[lay_index][:, grid_y, grid_x, box_index, 5 + label] = 1


def normalize(image):
    return image / 255.


if __name__ == '__main__':
    from configs.task_config import *

    true = create_empty_y_true(1, (52, 26, 13), 2)
    config_file = r"../configs\pascal_voc.json"
    with open(config_file) as data_file:
        config = json.load(data_file)

    model_cfg = ModelConfig(config["model"])
    train_cfg = TrainConfig(config["train"])
    train_generator = BatchGenerator(model_cfg, train_cfg, True)
    x, y = train_generator.get_next_batch()
    print(x[0])
    print(y[0].shape, y[1].shape, y[2].shape)

import glob
import os

from Config.train_config import *
from DataSet.image_enhance import ImageEnhance
from DataSet.pascalvoc_parser import PascalVocParser
from Utils.anchor_boxes import convert_to_encode_box
from Utils.bound_box import *
from Utils.convert import *


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

        self.annot_folder = train_configs.train_annot_folder if train_batch \
            else train_configs.valid_annot_folder
        self.img_dir = train_configs.train_image_folder if train_batch \
            else train_configs.valid_image_folder
        self.annot_filenames = self.get_ann_filenames() if train_batch \
            else self.get_ann_filenames()[0:train_configs.valid_size]

        self.learning_rate = train_configs.learning_rate
        self.enhance = train_configs.enhance
        self.shuffle = train_configs.shuffle
        self.batch_size = batch_size = train_configs.batch_size
        self.steps_per_epoch = int(len(self.annot_filenames) / batch_size)

        self.data_length = len(self.annot_filenames)
        self.save_folder = train_configs.save_folder
        self.epoch = train_configs.num_epoch

    def get_ann_filenames(self):
        return glob.glob(os.path.join(self.annot_folder, "*.xml"))

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            info += "{}:\t{}\r\n".format(key, value)
        return info

    def yield_next_batch(self):
        """
        get x_input (x image) and y_outputs (y true)
        """
        dataset_len = len(self.annot_filenames)
        i = 0
        while True:
            # x_inputs shape: [n,416,416,3]
            x_inputs = np.zeros((self.batch_size, self.input_size, self.input_size, 3))
            # y_outputs shape: [n,52,52,3,25],[n,26,26,3,25],,[n,13,13,3,25]
            y_outputs = [
                np.zeros((self.batch_size, self.pattern_shape[i], self.pattern_shape[i], 3, 5 + self.classes)).astype(
                    np.float32) for i in range(3)]

            # insert batch size of datas to x and y
            for batch_index in range(self.batch_size):
                # 随机打乱标签文件名列表
                # if i == 0:
                #     np.random.shuffle(self.annot_filenames)

                # step 1: initial annotation
                annotation = self.get_annotation(self.annot_filenames[i], self.img_dir, self.label_names)
                image_file, boxes, labels_code = annotation.image_filename, annotation.boxes, annotation.labels_code

                # step 2: initial x_inputs and update boxes
                x_inputs[batch_index, ...], boxes = self.get_image_with_enhance(image_file, boxes)

                # step 3: initial y_inputs
                for original_box, label in zip(boxes, labels_code):
                    match_index, match_anchor = self.get_match_anchor_boxes(original_box, self.anchors_boxes)
                    lay_index, box_index = match_index // 3, match_index % 3

                    code_box = convert_to_encode_box(self.pattern_shape[lay_index], self.input_size, original_box,
                                                     match_anchor)
                    self.assign_box(y_outputs, batch_index, lay_index, box_index, code_box, label)

                i = (i + 1) % dataset_len

            outputs = [y_outputs[0], y_outputs[1], y_outputs[2]]
            yield x_inputs, outputs

    def return_next_batch(self):
        """
        get x_input (x image) and y_outputs (y true)
        """
        dataset_len = len(self.annot_filenames)
        i = 0
        while True:
            # x_inputs shape: [n,416,416,3]
            x_inputs = np.zeros((self.batch_size, self.input_size, self.input_size, 3))
            # y_outputs shape: [n,52,52,3,25],[n,26,26,3,25],,[n,13,13,3,25]
            y_outputs = [
                np.zeros((self.batch_size, self.pattern_shape[i], self.pattern_shape[i], 3,
                          5 + self.classes)).astype(
                    np.float32) for i in range(3)]

            # insert batch size of datas to x and y
            for batch_index in range(self.batch_size):
                # 随机打乱标签文件名列表
                if i == 0:
                    np.random.shuffle(self.annot_filenames)

                # step 1: initial annotation
                annotation = self.get_annotation(self.annot_filenames[i], self.img_dir, self.label_names)
                image_file, boxes, labels_code = annotation.image_filename, annotation.boxes, annotation.labels_code

                # step 2: initial x_inputs and update boxes
                x_inputs[batch_index, ...], resize_boxes = self.get_image_with_enhance(image_file, boxes)

                # step 3: initial y_inputs
                for original_box, label in zip(resize_boxes, labels_code):
                    match_index, match_anchor = self.get_match_anchor_boxes(original_box, self.anchors_boxes)
                    lay_index, box_index = match_index // 3, match_index % 3

                    code_box = convert_to_encode_box(self.pattern_shape[lay_index], self.input_size,
                                                     original_box,
                                                     match_anchor)
                    self.assign_box(y_outputs, batch_index, lay_index, box_index, code_box, label)

                i = (i + 1) % dataset_len

            outputs = [y_outputs[0], y_outputs[1], y_outputs[2]]
            return x_inputs, outputs

    def get_image_with_enhance(self, image_file, boxes):
        """
        get image and update boxes when enable enhance
        :param image_file:
        :param boxes:
        :return:
        """
        img_augmenter = ImageEnhance(self.input_size, self.input_size, self.enhance)
        img, boxes = img_augmenter.get_image(image_file, boxes)
        return img / 255., boxes

    @staticmethod
    def get_annotation(ann_filename, image_dire, labels):
        parser = PascalVocParser(image_dire, labels)
        annotation = parser.get_annotation(ann_filename)
        return annotation

    @staticmethod
    def get_match_anchor_boxes(box, anchor_boxes):
        box = convert_to_centroid(np.array([box]))[0]
        bound_box = BoundBox(box[0], box[1], box[2], box[3])
        match_index, match_anchor = bound_box.get_match_anchor_box(anchor_boxes)
        return match_index, match_anchor

    @staticmethod
    def assign_box(yolo_true_output, batch_index, lay_index, box_index, box, label):
        _, _, _, _, grid_x, grid_y = box

        # assign ground truth x, y, w, h, confidence and class probability to y_batch
        yolo_true_output[lay_index][batch_index, grid_y, grid_x, box_index, 0:4] = box[0:4]
        yolo_true_output[lay_index][batch_index, grid_y, grid_x, box_index, 4] = 1.
        yolo_true_output[lay_index][batch_index, grid_y, grid_x, box_index, 5 + label] = 1


if __name__ == '__main__':
    config_file = r"..\config\raccoon.json"
    with open(config_file) as data_file:
        config = json.load(data_file)

    model_cfg = ModelConfig(config["model"])
    train_cfg = TrainConfig(config["train"])

    train_generator = BatchGenerator(model_cfg, train_cfg, True)

    x, y = train_generator.return_next_batch()
    print(y[1])

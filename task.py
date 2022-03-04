import os
from enum import Enum

from Config.train_config import *
from DataSet.batch_generator import BatchGenerator
from Nets.front_net import YoloDetector
from Nets.DarkNet.weight_reader import WeightReader
from Nets.yolo3_net import get_yolo3_backend
from Utils.utils import download_if_not_exists


class ModelInit(Enum):
    original = 1
    pretrain = 2
    random = 3


class TaskParser(object):

    def __init__(self, config_file):
        with open(config_file) as json_data:
            config = json.load(json_data)

        self.model_config = ModelConfig(config["model"])
        self.train_config = TrainConfig(config["train"])

    def set_train_config(self, train):
        self.train_config = train

    def get_train_config(self):
        return self.train_config

    def set_model_config(self, model):
        self.model_config = model

    def get_model_config(self):
        return self.model_config

    train_cfg = property(get_train_config, set_train_config)
    model_cfg = property(get_model_config, set_model_config)

    def create_model(self, model_init, skip_detect_layer=False):
        # model = YoloNet(self.model_cfg.classes)
        model = get_yolo3_backend((self.model_cfg.input_size, self.model_cfg.input_size), self.model_cfg.classes)
        keras_weights = self.train_config.pretrain_weight
        darknet_weight = self.train_config.darknet_weight

        if model_init == ModelInit.original:
            download_if_not_exists(darknet_weight, "https://pjreddie.com/media/files/yolov3.weights")
            weights_reader = WeightReader(darknet_weight)
            for block_model in model.layers:
                if block_model.name == "input_layer":
                    continue
                weights_reader.load_weights(block_model, skip_detect_layer)
            print("Original yolov3 weights loaded!!")
            return model

        elif model_init == ModelInit.pretrain and os.path.exists(keras_weights):
            model.load_weights(keras_weights)
            print("Keras pretrained weights loaded from {}!!".format(keras_weights))

        return model

    def create_generator(self):
        train_generator = BatchGenerator(self.model_cfg, self.train_cfg, True)
        valid_generator = BatchGenerator(self.model_cfg, self.train_cfg, False)

        print("Training samples : {}, Validation samples : {}".format(train_generator.data_length,
                                                                      valid_generator.data_length))
        return train_generator, valid_generator

    def create_detector(self, model, object_thresh=0.5, nms_thresh=0.5):
        return YoloDetector(model,
                            anchors=self.model_config.anchor_array,
                            image_size=self.model_config.input_size,
                            obj_thresh=object_thresh,
                            nms_thresh=nms_thresh)

    def create_evaluator(self, model):
        pass


if __name__ == "__main__":
    task = TaskParser("config/pascal_voc.json")
    print(task.train_cfg.__dict__)
    print(task.model_cfg.__dict__)
    net = task.create_model(ModelInit.original)
    net.summary()
    # test_x = tf.ones(shape=(1, 416, 416, 3), dtype=float)
    # test_y = net.predict(test_x)
    # print(test_y[0])
    # print(test_y[1])
    # print(test_y[2])

import json

from Utils.anchor_boxes import *


class ModelConfig(object):
    def __init__(self, config):
        self.labels = config["labels"]
        self.anchors = config["anchors"]
        self.input_size = config["input_size"]
        self.pattern_shape = config["pattern_shape"]
        self.anchor_array = create_anchor_array(self.anchors)
        self.anchor_boxes = create_anchor_boxes(self.anchors)
        self.classes = len(self.labels)
        self.input_shape = (self.input_size, self.input_size)

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            info += "{}:\t{}\r\n".format(key, value)
        return info


class TrainConfig(object):

    def __init__(self, config):
        self.num_epoch = config["num_epoch"]
        self.train_image_folder = config["train_image_folder"]
        self.train_annot_folder = config["train_annot_folder"]
        self.valid_image_folder = config["valid_image_folder"]
        self.valid_annot_folder = config["valid_annot_folder"]
        self.valid_size = config["valid_size"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.save_folder = config["save_folder"]
        self.enhance = config["enhance"]
        self.shuffle = config["shuffle"]
        self.random_net_size = config["random_net_size"]
        self.pretrain_weight = config["pretrain_weight"]
        self.darknet_weight = config["darknet_weight"]
        self.pre_train = config["pre_train"]

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            info += "{}:\t{}\r\n".format(key, value)
        return info


if __name__ == "__main__":
    with open("pascalVoc.json") as data_file:
        configs = json.load(data_file)

    model_config = ModelConfig(configs["model"])
    train_config = TrainConfig(configs["train"])

    print(model_config)
    print(train_config)

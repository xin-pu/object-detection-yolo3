# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from Nets.body_net import BodyNet
from Nets.head_net import HeadNet


# Will return DarkNet53 as Model
# It has 54 convolutional layers
# Input Shape:  [4,416,416,3]
# Num Classes:  20
# Output:       [[4,13,13,75],[4,26,26,75],[4,52,52,75]]
# 75 = 3 (different anchor_kmeans) * (5 (X,Y,W,H,P) + 20 classes)


def get_yolo3_backend(input_shape, num_classes, print_summary=False):
    model_input = Input(shape=(*input_shape, 3), dtype=float, name="input_layer")
    model_output = HeadNet(num_classes)(BodyNet()(model_input))
    model = Model(inputs=model_input, outputs=model_output, name="yolov3")
    if print_summary:
        # model.summary()
        print("Layers Number:   {0}".format(len(model.layers)))
    return model


if __name__ == '__main__':
    # (1, 256, 256, 3) => (1, 8, 8, 1024)
    inputs = tf.ones(shape=(4, 416, 416, 3))
    yolo_net = get_yolo3_backend((416, 416), 20, True)
    y = yolo_net(inputs)
    print(y[0].shape, y[1].shape, y[2].shape)

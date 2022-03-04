import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

# Darknet53 feature extractor
from Nets.Layer.convolutional_conv import ConvolutionalConv
from Nets.Layer.convolutional_five import Convolutional5
from Nets.Layer.convolutional_unsamling import ConvolutionalUnSampling


class HeadNet(Model):

    def __init__(self, n_classes=20):
        super(HeadNet, self).__init__(name='')
        self.classes = n_classes
        n_features: int = 3 * (n_classes + 5)

        self.stage5_conv5 = Convolutional5(512, [75, 76, 77, 78, 79])
        self.stage5_conv2 = ConvolutionalConv(512 * 2, n_features,
                                              [80, 81],
                                              name="detection_layer_1_{}".format(n_features))

        self.stage5_up_sampling = ConvolutionalUnSampling(512 / 2, [84])
        self.stage5_concatenate = Concatenate()

        self.stage4_conv5 = Convolutional5(256, [87, 88, 89, 90, 91])
        self.stage4_conv2 = ConvolutionalConv(256 * 2, n_features,
                                              [92, 93],
                                              name="detection_layer_2_{}".format(n_features))

        self.stage4_up_sampling = ConvolutionalUnSampling(256 / 2, [96])
        self.stage4_concatenate = Concatenate()

        self.stage3_conv5 = Convolutional5(128, [99, 100, 101, 102, 103])
        self.stage3_conv2 = ConvolutionalConv(128 * 2, n_features, [104, 105],
                                              name="detection_layer_3_{}".format(n_features))

        self.num_layers = 106

    def call(self, inputs, training=False, **kwargs):
        stage3_in, stage4_in, stage5_in = inputs
        x1 = self.stage5_conv5(stage5_in, training)
        y1 = self.stage5_conv2(x1, training)

        x = self.stage5_up_sampling(x1, training)
        x = self.stage5_concatenate([x, stage4_in])

        x2 = self.stage4_conv5(x, training)
        y2 = self.stage4_conv2(x2, training)

        x = self.stage4_up_sampling(x2, training)
        x = self.stage4_concatenate([x, stage3_in])

        x3 = self.stage3_conv5(x, training)
        y3 = self.stage3_conv2(x3, training)
        return [y3, y2, y1]

    def get_variables(self, layer_idx, suffix=None):
        if suffix:
            find_name = "layer_{}/{}".format(layer_idx, suffix)
        else:
            find_name = "layer_{}/".format(layer_idx)
        variables = []
        for var in self.variables:
            if find_name in var.name:
                variables.append(var)
        return variables

    def get_config(self):
        config = super(HeadNet, self).get_config()
        config.update({"classes": self.classes,
                       "layer_idx": self.layer_index})
        return config


if __name__ == '__main__':
    import numpy as np

    s3 = tf.constant(np.random.randn(1, 52, 52, 256).astype(np.float32))
    s4 = tf.constant(np.random.randn(1, 26, 26, 512).astype(np.float32))
    s5 = tf.constant(np.random.randn(1, 13, 13, 1024).astype(np.float32))

    headNet = HeadNet()
    f5, f4, f3 = headNet([s3, s4, s5])
    print(f5.shape, f4.shape, f3.shape)

    for v in headNet.variables:
        print(v.name)
    print("-" * 30)
    for lay in headNet.layers:
        print(lay.name)

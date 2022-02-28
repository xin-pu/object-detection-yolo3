# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *

# ---------------------------------------------------#
#   darknet53 的主体部分
#   输入为一张416x416x3的图片
#   输出为三个有效特征层
#   [Stage3]: 小尺度特征层 52*52
#   [Stage4]: 中尺度特征层 26*26
#   [Stage5]: 大尺度特征层 13*13
# ---------------------------------------------------#
from Nets.Layer.convolutional import Convolutional
from Nets.Layer.convolutional_pool import ConvolutionalPool
from Nets.Layer.convolutional_residual import ConvolutionalResidual


class BodyNet(Model):

    def __init__(self):
        super(BodyNet, self).__init__(name='')

        # Input: (256, 256, 3)
        self.l0a = Convolutional(32, layer_idx=0)

        #  (208, 208, 64)
        self.l0_pool = ConvolutionalPool(64, layer_idx=1)
        self.l1a = ConvolutionalResidual(64, layer_idx=[2, 3])

        # (104, 104, 128)
        self.l1_pool = ConvolutionalPool(128, layer_idx=4)
        self.l2a = ConvolutionalResidual(128, layer_idx=[5, 6])
        self.l2b = ConvolutionalResidual(128, layer_idx=[7, 8])

        # (52, 52, 256)s
        self.l2_pool = ConvolutionalPool(256, layer_idx=9)
        self.l3a = ConvolutionalResidual(256, layer_idx=[10, 11])
        self.l3b = ConvolutionalResidual(256, layer_idx=[12, 13])
        self.l3c = ConvolutionalResidual(256, layer_idx=[14, 15])
        self.l3d = ConvolutionalResidual(256, layer_idx=[16, 17])
        self.l3e = ConvolutionalResidual(256, layer_idx=[18, 19])
        self.l3f = ConvolutionalResidual(256, layer_idx=[20, 21])
        self.l3g = ConvolutionalResidual(256, layer_idx=[22, 23])
        self.l3h = ConvolutionalResidual(256, layer_idx=[24, 25])

        # (26, 26, 512)
        self.l3_pool = ConvolutionalPool(512, layer_idx=26)
        self.l4a = ConvolutionalResidual(512, layer_idx=[27, 28])
        self.l4b = ConvolutionalResidual(512, layer_idx=[29, 30])
        self.l4c = ConvolutionalResidual(512, layer_idx=[31, 32])
        self.l4d = ConvolutionalResidual(512, layer_idx=[33, 34])
        self.l4e = ConvolutionalResidual(512, layer_idx=[35, 36])
        self.l4f = ConvolutionalResidual(512, layer_idx=[37, 38])
        self.l4g = ConvolutionalResidual(512, layer_idx=[39, 40])
        self.l4h = ConvolutionalResidual(512, layer_idx=[41, 42])

        # (13, 13, 1024)
        self.l4_pool = ConvolutionalPool(1024, layer_idx=43)
        self.l5a = ConvolutionalResidual(1024, layer_idx=[44, 45])
        self.l5b = ConvolutionalResidual(1024, layer_idx=[46, 47])
        self.l5c = ConvolutionalResidual(1024, layer_idx=[48, 49])
        self.l5d = ConvolutionalResidual(1024, layer_idx=[50, 51])

        self.num_layers = 52

    def call(self, inputs, training=False, **kwargs):
        x = self.l0a(inputs, training)
        x = self.l0_pool(x, training)

        x = self.l1a(x, training)
        x = self.l1_pool(x, training)

        x = self.l2a(x, training)
        x = self.l2b(x, training)
        x = self.l2_pool(x, training)

        x = self.l3a(x, training)
        x = self.l3b(x, training)
        x = self.l3c(x, training)
        x = self.l3d(x, training)
        x = self.l3e(x, training)
        x = self.l3f(x, training)
        x = self.l3g(x, training)
        x = self.l3h(x, training)
        output_stage3 = x
        x = self.l3_pool(x, training)

        x = self.l4a(x, training)
        x = self.l4b(x, training)
        x = self.l4c(x, training)
        x = self.l4d(x, training)
        x = self.l4e(x, training)
        x = self.l4f(x, training)
        x = self.l4g(x, training)
        x = self.l4h(x, training)
        output_stage4 = x
        x = self.l4_pool(x, training)

        x = self.l5a(x, training)
        x = self.l5b(x, training)
        x = self.l5c(x, training)
        x = self.l5d(x, training)
        output_stage5 = x
        return [output_stage3, output_stage4, output_stage5]

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
        config = super(BodyNet, self).get_config()
        return config


if __name__ == '__main__':
    image = np.random.randn(1, 416, 416, 3).astype(np.float32)
    input_tensor = tf.constant(image)

    # (1, 256, 256, 3) => (1, 8, 8, 1024)
    body_net = BodyNet()
    s3, s4, s5 = body_net(input_tensor)

    print(s3.shape, s4.shape, s5.shape)

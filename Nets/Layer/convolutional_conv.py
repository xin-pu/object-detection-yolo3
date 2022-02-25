
from tensorflow.keras.layers import *

from Nets.Layer.convolutional import Convolutional


# ---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数 + 卷积
#   DarknetConv2D + BatchNormalization + LeakyReLU + DarknetConv2D
# ---------------------------------------------------#
class ConvolutionalConv(Layer):
    def __init__(self, filters, out_filters, layer_idx, name=""):
        super(ConvolutionalConv, self).__init__(name=name)

        layer_names = ["layer_{}".format(i) for i in layer_idx]

        self.conv = Convolutional(filters, layer_idx[0], (3, 3), name=layer_names[0])
        self.conv2 = Conv2D(out_filters, (1, 1), strides=(1, 1),
                            padding='same',
                            use_bias=True,
                            name=layer_names[1])

    def call(self, inputs, training=False, **kwargs):
        x = self.conv(inputs)
        x = self.conv2(x)
        return x


if __name__ == "__main__":
    import tensorflow as tf
    ones = tf.ones(shape=(1, 416, 416, 3))

    con = Convolutional(32, 1)
    res = con.call(ones, True)
    print(res)

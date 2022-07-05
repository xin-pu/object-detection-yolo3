from tensorflow.keras.layers import *
from tensorflow.python.keras.regularizers import l1_l2
from tensorflow.python.ops.initializers_ns import random_normal


# ---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#
class Convolutional(Layer):
    def __init__(self,
                 filters,
                 layer_idx,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 name=""):
        super(Convolutional, self).__init__(name=name)

        layer_name = "layer_{}".format(str(layer_idx))

        self.conv = Conv2D(filters, kernel_size,
                           strides=strides,
                           padding=padding,
                           use_bias=False,
                           kernel_initializer=random_normal(stddev=0.02),
                           # kernel_regularizer=l1_l2(5e-4),
                           name=layer_name)
        self.bn = BatchNormalization(name=layer_name)
        self.leaky_relu = LeakyReLU(alpha=0.1, name=layer_name)

    def call(self, inputs, training=False, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.leaky_relu(x)
        return x


if __name__ == "__main__":
    import tensorflow as tf
    ones = tf.ones(shape=(1, 416, 416, 3))

    con = Convolutional(32, 1)
    res = con.call(ones, True)
    print(res)

from tensorflow.keras.layers import *

from Nets.Layer.convolutional import Convolutional


class ConvolutionalPool(Layer):
    def __init__(self, filters, layer_idx, name=""):
        super(ConvolutionalPool, self).__init__(name=name)

        layer_name = "layer_{}".format(layer_idx)

        self.pad = ZeroPadding2D(((1, 0), (1, 0)))
        self.conv2 = Convolutional(filters, layer_idx, (3, 3), strides=(2, 2), padding='valid',
                                   name=layer_name)

    def call(self, inputs, training=False, **kwargs):

        x = self.pad(inputs)
        x = self.conv2(x, training)

        return x


if __name__ == "__main__":
    import tensorflow as tf

    ones = tf.ones(shape=(1, 416, 416, 3))

    con = ConvolutionalPool(32, 1)
    res = con.call(ones, True)
    print(res)

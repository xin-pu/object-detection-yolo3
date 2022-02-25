from tensorflow.keras.layers import *

from Nets.Layer.convolutional import Convolutional


class Convolutional5(Layer):
    def __init__(self, filters, layer_idx, name=""):
        super(Convolutional5, self).__init__(name=name)
        self.conv1 = Convolutional(filters, layer_idx[0], kernel_size=(1, 1))
        self.conv2 = Convolutional(filters * 2, layer_idx[1], kernel_size=(3, 3))
        self.conv3 = Convolutional(filters, layer_idx[2], kernel_size=(1, 1))
        self.conv4 = Convolutional(filters * 2, layer_idx[3], kernel_size=(3, 3))
        self.conv5 = Convolutional(filters, layer_idx[4], kernel_size=(1, 1))

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs, training)
        x = self.conv2(x, training)
        x = self.conv3(x, training)
        x = self.conv4(x, training)
        x = self.conv5(x, training)
        return x


if __name__ == "__main__":
    import tensorflow as tf

    ones = tf.ones(shape=(1, 416, 416, 3))
    con = Convolutional5(32, [1, 2, 3, 4, 5])
    res = con.call(ones, True)
    print(res)


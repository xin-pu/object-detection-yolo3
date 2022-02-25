from tensorflow.keras.layers import *

from Nets.Layer.convolutional import Convolutional


class ConvolutionalUnSampling(Layer):
    def __init__(self, filters, layer_idx, name=""):
        super(ConvolutionalUnSampling, self).__init__(name=name)

        layer_names = ["layer_{}".format(i) for i in layer_idx]

        self.conv = Convolutional(filters, layer_idx=layer_idx[0], kernel_size=(1, 1), name=layer_names[0])
        self.un_sampling = UpSampling2D(2)

    def call(self, inputs, training=False, **kwargs):
        x = self.conv(inputs, training)
        x = self.un_sampling(x)
        return x


if __name__ == "__main__":
    import tensorflow as tf

    ones = tf.ones(shape=(1, 416, 416, 3))

    con = ConvolutionalUnSampling(32, [1])
    res = con.call(ones, True)
    print(res)

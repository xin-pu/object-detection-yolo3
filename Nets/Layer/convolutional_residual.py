from tensorflow.keras.layers import *


class ConvolutionalResidual(Layer):
    def __init__(self, filters, layer_idx, name=""):
        super(ConvolutionalResidual, self).__init__(name=name)
        filters1, filters2 = filters / 2, filters
        layer1, layer2 = layer_idx

        layer_name1 = "layer_{}".format(str(layer1))
        layer_name2 = "layer_{}".format(str(layer2))

        self.conv2a = Conv2D(filters1, (1, 1), padding='same', use_bias=False, name=layer_name1)
        self.bn2a = BatchNormalization(name=layer_name1)
        self.leaky_relu1 = LeakyReLU(alpha=0.1)

        self.conv2b = Conv2D(filters2, (3, 3), padding='same', use_bias=False, name=layer_name2)
        self.bn2b = BatchNormalization(name=layer_name2)
        self.leaky_relu2 = LeakyReLU(alpha=0.1)

        self.add = Add()

    def call(self, inputs, training=False, **kwargs):
        x = self.conv2a(inputs)
        x = self.bn2a(x, training=training)
        x = self.leaky_relu1(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = self.leaky_relu2(x)

        x = self.add([inputs, x])
        return x


if __name__ == "__main__":
    import tensorflow as tf

    ones = tf.ones(shape=(1, 416, 416, 16))

    con = ConvolutionalResidual(16, [1, 2])
    res = con.call(ones, True)
    print(res)

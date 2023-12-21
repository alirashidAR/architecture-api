import numpy as np
from tensorflow import keras
from keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
    Activation,
    Concatenate,
    Dropout,
    BatchNormalization,
)
from keras.initializers import RandomNormal
from keras.models import Model


class ImageGenModel:
    def __init__(self, model_path):
        self.model = self.define_generator((256, 256, 3))
        self.model.load_weights(model_path)

    def generate(self, input_images, verbose=0):
        X = np.array(input_images).reshape(-1, 256, 256, 3) / 255.0
        result = self.model.predict(X, verbose=verbose)

        return result

    def define_encoder_block(self, layer_in, n_filters, batchnorm=True):
        init = RandomNormal(stddev=0.02, seed=1)

        g = Conv2D(
            n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(layer_in)
        if batchnorm:
            g = BatchNormalization()(g, training=True)
        g = LeakyReLU(alpha=0.2)(g)

        return g

    def define_decoder_block(self, layer_in, skip_in, n_filters, dropout=True):
        init = RandomNormal(stddev=0.02, seed=1)

        g = Conv2DTranspose(
            n_filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(layer_in)
        g = BatchNormalization()(g, training=True)
        if dropout:
            g = Dropout(0.5)(g, training=True)
        g = Concatenate()([g, skip_in])
        g = Activation("relu")(g)

        return g

    def define_generator(self, image_shape=(256, 256, 3)):
        init = RandomNormal(stddev=0.02, seed=1)

        in_image = Input(shape=image_shape)

        e1 = self.define_encoder_block(in_image, 64, batchnorm=False)
        e2 = self.define_encoder_block(e1, 128)
        e3 = self.define_encoder_block(e2, 256)
        e4 = self.define_encoder_block(e3, 512)
        e5 = self.define_encoder_block(e4, 512)
        e6 = self.define_encoder_block(e5, 512)
        e7 = self.define_encoder_block(e6, 512)

        b = Conv2D(
            512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init
        )(e7)
        b = Activation("relu")(b)

        d1 = self.define_decoder_block(b, e7, 512)
        d2 = self.define_decoder_block(d1, e6, 512)
        d3 = self.define_decoder_block(d2, e5, 512)
        d4 = self.define_decoder_block(d3, e4, 512, dropout=False)
        d5 = self.define_decoder_block(d4, e3, 256, dropout=False)
        d6 = self.define_decoder_block(d5, e2, 128, dropout=False)
        d7 = self.define_decoder_block(d6, e1, 64, dropout=False)

        g = Conv2DTranspose(
            image_shape[2],
            (4, 4),
            strides=(2, 2),
            padding="same",
            kernel_initializer=init,
        )(d7)
        out_image = Activation("tanh")(g)

        model = Model(in_image, out_image)

        return model

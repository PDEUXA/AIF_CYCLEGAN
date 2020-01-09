import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl


# Define Generator architecture

# Resnet Block, (Unet)
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    # first layer convolutional layer
    r = kl.Conv2D(256, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    r = kl.Activation('relu')(r)

    # second convolutional layer
    r = kl.Conv2D(256, (3, 3), padding='same', kernel_initializer=init)(r)

    # concatenate merge channel-wise with input layer
    r = kl.Concatenate()([r, input_layer])

    return r


def generator(image_shape):
    # weight initialization
    init = tf.keras.initializers.RandomNormal(stddev=0.02)

    input_image = tf.keras.Input(shape=image_shape)

    # Encoding part 64-->128-->256 (downsampling)
    g = kl.Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(input_image)
    g = kl.Activation('relu')(g)
    g = kl.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = kl.Activation('relu')(g)
    g = kl.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = kl.Activation('relu')(g)

    # Resnet Block, (Unet)
    for _ in range(9):
        g = resnet_block(256, g)

    # Decoding part 256-->128-->64-->3
    g = kl.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = kl.Activation('relu')(g)
    g = kl.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
    g = kl.Activation('relu')(g)
    g = kl.Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(g)

    out = kl.Activation('tanh')(g)
    gen = km.Model(input_image, out)

    return gen

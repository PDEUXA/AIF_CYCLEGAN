import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko

# Define Discriminator architecture


def discriminator(image_shape):

    # weight initialization
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    # source image input
    disc = km.Sequential()
    disc.add(kl.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    disc.add(kl.LeakyReLU(alpha=0.2))
    disc.add(kl.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    disc.add(kl.LeakyReLU(alpha=0.2))
    # C256
    disc.add(kl.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    disc.add(kl.LeakyReLU(alpha=0.2))
    # C512
    disc.add(kl.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init))
    disc.add(kl.LeakyReLU(alpha=0.2))
    disc.add(kl.Conv2D(512, (4, 4), padding='same', kernel_initializer=init))
    disc.add(kl.LeakyReLU(alpha=0.2))

    # patch output
    disc.add(kl.Conv2D(1, (4, 4), padding='same', kernel_initializer=init))
    disc.compile(loss='mse', optimizer=ko.Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return disc

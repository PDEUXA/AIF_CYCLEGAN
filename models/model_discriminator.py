import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
#import tensorflow_addons as tfa
from models.InstanceNormalize import *
# Define Discriminator architecture


def normes(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return kl.BatchNormalization
    elif norm == 'instance_norm':
        return InstanceNormalization
    elif norm == 'layer_norm':
        return kl.LayerNormalization


def discriminator(input_shape=(256, 256, 3), dim=64, n_downsamplings=3, norm='instance_norm'):
    dim_ = dim
    norme = normes(norm)

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    h = kl.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = kl.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = norme()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = kl.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = norme()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = kl.Conv2D(1, 4, strides=1, padding='same')(h)

    return km.Model(inputs=inputs, outputs=h)

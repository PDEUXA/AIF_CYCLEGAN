import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow_addons as tfa


# Define Generator architecture
def normes(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return kl.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return kl.LayerNormalization


def generator(input_shape=(256, 256, 3), output_channels=3, dim=64, n_downsamplings=2, n_blocks=9, norm='instance_norm'):
    norme = normes(norm)

    def resnet_block(x):
        dim = x.shape[-1]
        h = x

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = kl.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = norme()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = kl.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = norme()(h)

        return kl.add([x, h])

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1, Convolution 64 filtres
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = kl.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = norme()(h)
    h = tf.nn.relu(h)

    # 2, Convolution 128, et 256 filtres
    for _ in range(n_downsamplings):
        dim *= 2
        h = kl.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = norme()(h)
        h = tf.nn.relu(h)

    # 3, Resnet Block (x9)
    for _ in range(n_blocks):
        h = resnet_block(h)

    # 4, Déconvolution 128, 64 filtres
    for _ in range(n_downsamplings):
        dim //= 2
        h = kl.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = norme()(h)
        h = tf.nn.relu(h)

    # 5, Convolution 3 filtres
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = kl.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)

    return km.Model(inputs=inputs, outputs=h)

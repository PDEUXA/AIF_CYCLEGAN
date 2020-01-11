import tensorflow as tf


def normalize(image):
    # normalizing the images to [-1, 1]
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def preprocess_image(image,label):
    image = normalize(image)
    return image


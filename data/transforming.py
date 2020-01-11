import tensorflow as tf


def normalize(image):
    # normalizing the images to [-1, 1]
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def preprocess_image_train(image,label):
    image = normalize(image)
    return image


def preprocess_image_test(image,label):
    image = normalize(image)
    return image

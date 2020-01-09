import tensorflow as tf


# Define losses
LAMBDA = 10  # Additional Weigh for the cycle loss and identity loss

# Generator loss


def gen_loss(generated):
    # Maximise the likehood of generated photo to be considered real, ie 1
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(generated), generated)


# Discriminator Loss


def disc_loss(real, generated):
    # Maximise the likehood of the real photo, ie 1
    # Minimise the likehood of generated photo, ie 0
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real), real)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss/2


# Cycle loss


def cycle_loss(real_image, cycled_image):
    # difference between original image an cycled image
    cycl_loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * cycl_loss


# Identity loss


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

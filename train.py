from models.losses import *
import tensorflow as tf


@tf.function
def train_step(real_a, real_b, gen_a, gen_b, disc_a, disc_b):

    # Set optimizer for generators and discriminators
    generator_a_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_b_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_a_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_b_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generators
        fake_b = gen_a(real_a, training=True)
        cycled_a = gen_b(fake_b, training=True)
        fake_a = gen_b(real_b, training=True)
        cycled_b = gen_a(fake_a, training=True)

        # same_a and same_b are used for identity loss.
        same_a = gen_a(real_a, training=True)
        same_b = gen_b(real_b, training=True)

        # Discriminator
        disc_real_a = disc_a(real_a, training=True)
        disc_real_b = disc_b(real_b, training=True)
        disc_fake_a = disc_a(fake_a, training=True)
        disc_fake_b = disc_b(fake_b, training=True)

        # calculate the loss
        gen_a_loss = gen_loss(disc_fake_b)
        gen_b_loss = gen_loss(disc_fake_a)

        total_cycle_loss = cycle_loss(real_a, cycled_a) + cycle_loss(real_b, cycled_b)

        # Total generator loss = adversarial loss + cycle loss + identity loss
        total_gen_b_loss = gen_b_loss + total_cycle_loss + identity_loss(real_b, same_b)
        total_gen_a_loss = gen_a_loss + total_cycle_loss + identity_loss(real_a, same_a)

        disc_a_loss = disc_loss(disc_real_a, disc_fake_a)
        disc_b_loss = disc_loss(disc_real_b, disc_fake_b)

    # Calculate the gradients for generator and discriminator
    generator_b_gradients = tape.gradient(total_gen_b_loss, gen_b.trainable_variables)
    generator_a_gradients = tape.gradient(total_gen_a_loss, gen_a.trainable_variables)

    discriminator_a_gradients = tape.gradient(disc_a_loss, disc_a.trainable_variables)
    discriminator_b_gradients = tape.gradient(disc_b_loss, disc_b.trainable_variables)

    # Apply the gradients to the optimizer
    generator_b_optimizer.apply_gradients(zip(generator_b_gradients, gen_b.trainable_variables))
    generator_a_optimizer.apply_gradients(zip(generator_b_gradients, gen_b.trainable_variables))
    discriminator_a_optimizer.apply_gradients(zip(discriminator_a_gradients, disc_a.trainable_variables))
    discriminator_b_optimizer.apply_gradients(zip(discriminator_a_gradients, disc_a.trainable_variables))

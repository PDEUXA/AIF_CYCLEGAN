from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import time
from data.data_loader import *
from data.displaying import generate_and_save_images
from models.model_discriminator import *
from models.model_generator import *
from train import train_step
from data.transforming import preprocess_image_train, preprocess_image_test

# Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='ukiyoe2photo')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--cycle_loss_weight', type=float, default=10.0)
parser.add_argument('--identity_loss_weight', type=float, default=0)
args = parser.parse_args()

# Set Global Variables
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
tfds.disable_progress_bar()
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Load data
train_a, train_b, test_a, test_b = load_data(name='ukiyoe2photo')


# Transform data
train_a = train_a.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
train_b = train_b.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
test_a = test_a.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
test_b = test_b.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)

# Instanciate Networks
gen_a = generator((256, 256, 3))
gen_b = generator((256, 256, 3))
disc_a = discriminator((256, 256, 3))
disc_b = discriminator((256, 256, 3))

# Optimiseur
generator_a_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_b_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_a_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_b_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Save check
# Create a checkpoint
checkpoint_path = "./checkpoints"

ckpt = tf.train.Checkpoint(generator_g=gen_a,
                           generator_f=gen_b,
                           discriminator_x=disc_a,
                           discriminator_y=disc_b,
                           generator_g_optimizer=generator_a_optimizer,
                           generator_f_optimizer=generator_b_optimizer,
                           discriminator_x_optimizer=discriminator_a_optimizer,
                           discriminator_y_optimizer=discriminator_b_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# Draw a sample
sample_a = next(iter(train_a))
sample_b = next(iter(train_b))

# Train the models
seed = tf.random.normal([1, 256, 256, 3])
for epoch in range(10):
    start = time.time()
    n = 0
    for image_A, image_B in tf.data.Dataset.zip((train_a, train_b)):
        print("trainstep")
        train_step(image_A, image_B, gen_a, gen_b, disc_a, disc_b, generator_a_optimizer,
                   generator_b_optimizer, discriminator_a_optimizer, discriminator_b_optimizer)
        if n % 10 == 0:
            print(n / 10, end=' ')
        n += 1

    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    generate_and_save_images(gen_a, epoch + 1, sample_a, 'A')
    generate_and_save_images(gen_b, epoch + 1, sample_b, 'B')

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time() - start))
#

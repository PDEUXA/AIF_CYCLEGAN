import tensorflow_datasets as tfds


def load_data(name='ukiyoe2photo'):
    # load data from TensorFlow
    dataset, metadata = tfds.load('cycle_gan/' + name, with_info=True, as_supervised=True)

    train_a, train_b = dataset['trainA'], dataset['trainB']
    test_a, test_b = dataset['testA'], dataset['testB']

    return train_a, train_b, test_a, test_b

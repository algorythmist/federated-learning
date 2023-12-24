import keras
import numpy as np

from simplefql.client import FederatedClient


def build_simple_model():
    return keras.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Rescaling(scale=1 / 255),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])


def build_and_compile_simple_model():
    model = build_simple_model()
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer='sgd',
                  metrics=[keras.metrics.sparse_categorical_accuracy])
    return model


def shuffle_together(*arrays, seed=None):
    """
    Shuffle arrays in unison
    :param arrays: a tuple or list of arrays. They must all have the same length
    :return: a tuple of the shuffled arrays
    """
    if seed is not None:
        np.random.seed(seed)
    if len(arrays) == 0:
        return []
    p = np.random.permutation(len(arrays[0]))
    return (a[p] for a in arrays)


def split_data(images, labels, shards):
    # randomize the data
    images, labels = shuffle_together(images, labels)
    # determine the size of each shard
    size = len(labels) // shards
    return [(images[i:i + size], labels[i:i + size]) for i in range(0, size * shards, size)]


def create_clients(shards, initial='client', create_model_fn=build_and_compile_simple_model):
    num_clients = len(shards)
    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]
    return [FederatedClient(client_names[i], create_model_fn(), shards[i])
            for i in range(len(client_names))]

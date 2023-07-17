import keras
import numpy as np
from federated import ClientNode


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





def create_clients(images, labels, num_clients=10, initial='client'):
    def shuffle_together(a, b):
        p = np.random.permutation(len(a))
        return a[p], b[p]

    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # randomize the data
    images, labels = shuffle_together(images, labels)

    # shard data and place at each client
    size = len(labels) // num_clients
    shards = [(images[i:i + size], labels[i:i + size]) for i in range(0, size * num_clients, size)]

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    return [ClientNode(client_names[i], build_and_compile_simple_model(), shards[i])
            for i in range(len(client_names))]
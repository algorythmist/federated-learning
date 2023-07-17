import keras
from util import create_clients, build_and_compile_simple_model
from federated import CentralServer

if __name__ == '__main__':
    dataset = keras.datasets.fashion_mnist

    (X_train, y_train), (X_test, y_test) = dataset.load_data()
    clients = create_clients(X_train, y_train, 5)
    server = CentralServer(build_and_compile_simple_model(), clients)
    server.train(20)

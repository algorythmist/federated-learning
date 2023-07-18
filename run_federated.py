from util import *
from federated import CentralServer

if __name__ == '__main__':
    dataset = keras.datasets.fashion_mnist

    (X_train, y_train), (X_test, y_test) = dataset.load_data()

    # split the training data between the server and the clients
    n_clients = 5
    shards = split_data(X_train, y_train, n_clients + 1)
    server_X, server_y = shards[0]
    clients = create_clients(shards=shards[1:], create_model_fn=build_and_compile_simple_model)

    # create test and validation sets
    half = len(y_test) // 2
    X_valid, y_valid = X_test[:half], y_test[:half]
    X_test, y_test = X_test[half:], y_test[half:]

    server_model = build_and_compile_simple_model()
    # pre-train server model to obtain initial weights
    server_model.fit(server_X, server_y, epochs=10)

    server = CentralServer(server_model, clients, client_epochs=3)
    server.train(20, evaluate_fn=lambda model: model.evaluate(X_test, y_test) )
    loss, accuracy = server_model.evaluate(X_valid, y_valid)
    print(f" accuracy: {accuracy} | loss: {loss}")


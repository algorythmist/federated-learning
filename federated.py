import tensorflow as tf


class ClientNode:
    """
    A client node performs training using a copy of the shared model on its own datasets.
    The weights are reset by the server at the beginning of each round.
    """

    def __init__(self, name, model, dataset):
        self.name = name
        self.model = model
        self.train_X, self.train_y = dataset

    def train(self, weights, epochs=5):
        """
        Set the model weights and train for the given number of epochs.
        :param weights: The initial weights to set the model to.
        :param epochs: The number of epochs to train for.
        :return: The number of samples trained on and the updated model weights.
        """
        # TODO: train mini batches
        self.model.set_weights(weights)
        self.model.fit(self.train_X, self.train_y, epochs=epochs)
        return len(self.train_y), self.model.get_weights()


class CentralServer:
    """
    The central server coordinates training by aggregating model weights from clients.
    """

    def __init__(self, model, clients):
        self.model = model
        self.clients = clients

    def train(self, iterations=10, evaluate_fn=None):
        """
        Train the model for the given number of iterations.
        :param iterations: The number of iterations to train for.
        :param evaluate_fn: An optional function to evaluate the model after each iteration.
        :return:
        """
        for iteration in range(iterations):
            local_weights = self.training_step()
            scaled_weights = self.__scale_weights(local_weights.values())
            average_weights = self.__aggregate_scaled_weights(scaled_weights)
            # update global model
            self.model.set_weights(average_weights)
            if evaluate_fn:
                evaluate_fn(self.model)

    def __scale_weights(self, local_weights):
        total_samples = float(sum([size for size, _ in local_weights]))
        return [self.__scale_model_weights(size / total_samples, weights) for size, weights in local_weights]

    @staticmethod
    def __scale_model_weights(scalar, weights):
        return [scalar * w for w in weights]

    def __aggregate_scaled_weights(self, scaled_weights):
        """
        Return the sum of the listed scaled weights.
        The is equivalent to scaled average of the weights
        """
        average_weights = []
        # get the average grad across all client gradients
        for weight_list_tuple in zip(*scaled_weights):
            layer_mean = tf.math.reduce_sum(weight_list_tuple, axis=0)
            average_weights.append(layer_mean)
        return average_weights

    def training_step(self):
        global_weights = self.model.get_weights()
        local_weights = {}
        # TODO: shuffle clients
        for client in self.clients:
            print(f"Fitting local model for client {client.name}")
            local_weights[client.name] = client.train(global_weights, epochs=1)
        return local_weights
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
        self.model.fit(self.train_X, self.train_y, epochs=epochs, verbose=0)
        return len(self.train_y), self.model.get_weights()


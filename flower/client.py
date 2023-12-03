import warnings
from collections import OrderedDict
import flwr as fl
import torch
from tqdm import tqdm
from model import SimpleCNN
from cifar_data import load_data

# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, net):
        self.net = net

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self._train(trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self._test(testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

    def _train(self, trainloader, epochs):
        """Train the model on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        for _ in range(epochs):
            for images, labels in tqdm(trainloader):
                optimizer.zero_grad()
                criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
                optimizer.step()

    def _test(self, testloader):
        """Validate the model on the test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for images, labels in tqdm(testloader):
                outputs = self.net(images.to(DEVICE))
                labels = labels.to(DEVICE)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        return loss, accuracy


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SimpleCNN().to(DEVICE)
    trainloader, testloader = load_data()

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(net),
    )

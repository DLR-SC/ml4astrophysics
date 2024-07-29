"""
Pytorch implementation of a fully connected neural network and a convolutional neural network
"""
from torch import nn


class FullyConnectedNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.one_hidden_layer = nn.Sequential(
            nn.Linear(28 * 28, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        x = self.one_hidden_layer(x)
        return x


class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # Your Code ...

        # def forward(self, x):
        # Your Code

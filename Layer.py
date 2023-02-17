import math
from random import random


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def dsigmoid(x: float) -> float:
    return x * (1 - x)


class Layer:

    def __init__(self, inputSize:int, outputSize:int) -> None:
        self.neurons = []
        self.biases = []
        self.weights = []
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = sigmoid
        self.derivative = dsigmoid
        for i in range(outputSize):
            self.biases.append(0)
            self.weights.append([])
            for j in range(inputSize):
                self.weights[i].append(random() - 0.5)


class InputLayer():
    def __init__(self, size:int) -> None:
        self.neurons = []
        self.biases = []
        self.weights = []
        self.outputSize = size
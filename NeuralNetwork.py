from Layer import Layer


class NeuralNetwork:

    learningRate: float
    layers: list[Layer]

    def __init__(self, layers: list[Layer], learningRate: float = 0.01) -> None:
        self.learningRate = learningRate
        self.layers = layers

    def forward(self, inputs) -> list[float]:
        self.layers[0].neurons = inputs
        input = inputs
        for layer in self.layers[1:]:
            layer.neurons = []
            for i in range(layer.outputSize):
                out = 0
                for j in range(layer.inputSize):
                    out += layer.weights[i][j] * input[j]
                out += layer.biases[i]
                layer.neurons.append(layer.activation(out))
            input = layer.neurons
        return self.layers[-1].neurons

    def backpropagation(self, target: list[float]):
        # output - tagret
        errors = []

        # err * activation'(x)
        gradients = []

        for i in range(self.layers[-1].outputSize):
            errors.append(self.layers[-1].neurons[i] - target[i])
            gradients.append(
                errors[i] * self.layers[-1].derivative(self.layers[-1].neurons[i]))

        for l in range(len(self.layers) - 1, 0, -1):
            curLayer = self.layers[l]
            prevLayer = self.layers[l - 1]

            # prev weights - learningRate * gradient * input
            newWeights = []
            newBias = []
            for i in range(curLayer.outputSize):
                newWeights.append([])
                for j in range(curLayer.inputSize):
                    newWeights[i].append(
                        curLayer.weights[i][j] - self.learningRate * gradients[i] * prevLayer.neurons[j])
                newBias.append(curLayer.biases[i] -
                               self.learningRate * gradients[i])

            if (l != 1):
                # (prev gradients * weights) * activation'(x)
                newGradients = []
                for i in range(prevLayer.outputSize):
                    g = 0
                    for j in range(curLayer.outputSize):
                        g += gradients[j] * curLayer.weights[j][i]
                    newGradients.append(
                        g * prevLayer.derivative(prevLayer.neurons[i]))
                gradients = newGradients
                
            curLayer.weights = newWeights
            curLayer.biases = newBias
            

    def train(self, x: list[list[float]], y: list[list[float]], epochs: int = 1000) -> None:
        print("Start training")
        for epoch in range(epochs):
            if epoch % (epochs / 10) == 0:
                print(f'epoch {epoch}')
            for _x, _y in zip(x, y):
                self.forward(_x)
                self.backpropagation(_y)

    def predict(self, x: list[float]) -> list[float]:
        return self.forward(x)

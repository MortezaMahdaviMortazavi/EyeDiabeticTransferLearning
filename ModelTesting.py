import numpy


# Implement Neural Network from scratch

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = numpy.random.rand(self.input.shape[1], 4)  # considering we have 4 nodes in the hidden layer
        self.weights2 = numpy.random.rand(4, 1)
        self.y = y
        self.output = numpy.zeros(y.shape)

    def sigmoid(self,x):
        return 1 / (1 + numpy.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1 - x)
    

    def feedforward(self):
        self.layer1 = self.sigmoid(numpy.dot(self.input, self.weights1))
        self.output = self.sigmoid(numpy.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = numpy.dot(self.layer1.T, (2 * (self.y - self.output) * self.sigmoid_derivative(self.output)))
        d_weights1 = numpy.dot(self.input.T, (
                    numpy.dot(2 * (self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T) *
                    self.sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


    def fit(self, epochs):
        for i in range(epochs):
            self.feedforward()
            self.backprop()
        
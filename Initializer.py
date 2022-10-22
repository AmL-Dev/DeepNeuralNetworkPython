from Var import *
import random


class Initializer:
    """
    Abstract class to initialize weights and biases for a layer of n_in inputs and n_out outputs

    Attributes:
      n_in: number of inputs for the layer
      n_out: number of outputs for the layer
    """

    def init_weights(self, n_in, n_out):
        raise NotImplementedError

    def init_bias(self, n_out):
        raise NotImplementedError


class NormalInitializer(Initializer):
    """
    Initialize variables using a normal distribution 
    """

    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def init_weights(self, n_in, n_out):
        return [[Var(random.gauss(self.mean, self.std)) for _ in range(n_out)] for _ in range(n_in)]

    def init_bias(self, n_out):
        return [Var(0.0) for _ in range(n_out)]


class ConstantInitializer(Initializer):
    """
    Initialize variables using a constant value
    """

    def __init__(self, weight=1.0, bias=0.0):
        self.weight = weight
        self.bias = bias

    def init_weights(self, n_in, n_out):
        return [[Var(self.weight) for _ in range(n_out)] for _ in range(n_in)]

    def init_bias(self, n_out):
        return [Var(self.bias) for _ in range(n_out)]

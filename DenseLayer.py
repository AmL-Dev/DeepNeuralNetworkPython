from ast import Import
from typing import Sequence
from Initializer import *
from Var import *


class DenseLayer:
    """
    Represents a layer in the neural network, with n_in input neurons and n_out output neurons and anctivation function act_fn.
    Depending on the initializer that is passed in, the initial weights and biais will be generated differently

    Attributes:
      n_in: number of inputs for the layer
      n_out: number of outputs for the layer
      act_fn: the activation function used in the feedforward (lambda function)
      initializer: initializer type for weights and biases
    """

    def __init__(self, n_in: int, n_out: int, act_fn, initializer=NormalInitializer()):
        self.weights = initializer.init_weights(n_in, n_out)
        self.bias = initializer.init_bias(n_out)
        self.act_fn = act_fn

    def __repr__(self):
        return 'DenseLayer(Size: (' + str(len(self.weights)) + ', ' + str(len(self.weights[0])) + '); Weights: ' + repr(self.weights) + '; Biases: ' + repr(self.bias) + ')'

    # Returns the list of all weights and biases
    def parameters(self) -> Sequence[Var]:
        params = []
        for r in self.weights:
            params += r

        return params + self.bias

    # Computes the values of the output layer given an input vector
    def forward(self, single_input: Sequence[Var]) -> Sequence[Var]:
        # self.weights is a matrix with dimension n_in x n_out. We check that the dimensionality of the input
        # to the current layer matches the number of nodes in the current layer
        assert len(self.weights) == len(
            single_input), "weights and single_input must match in first dimension"
        weights = self.weights
        out = []
        # For some given data point single_input, we now want to calculate the resulting value in each node in the current layer
        # We therefore loop over the (number of) nodes in the current layer:
        for j in range(len(weights[0])):
            # Initialize the node value depending on its corresponding parameters.
            node = self.bias[j]
            # We now finish the linear transformation corresponding to the parameters of the currently considered node.
            for i in range(len(single_input)):
                node += single_input[i] * weights[i][j]
            node = self.act_fn(node)
            out.append(node)

        return out

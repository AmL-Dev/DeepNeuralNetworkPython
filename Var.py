# Inspired from https://github.com/rasmusbergpalm/nanograd/blob/3a1bf9e9e724da813bfccf91a6f309abdade9f39/nanograd.py

from math import exp, log, pow


class Var:
    """
    A variable which holds a float and enables gradient computations.

    Attributes:
      v: stores the value of the variable
      grad_fn: when applying a function to the variable involving another variable f(a,b) (i.e. addition, multiplication ...), this stores [(a, df/da), (b,df)]
               when applying a function to the variable only f(a) (power, exponential function ...), this stores [(a, df/da)
      grad: stores the value of the gradient of the variable (updated after calling backprop)
    """

    def __init__(self, val: float, grad_fn=lambda: []):
        assert type(val) == float
        self.v = val
        self.grad_fn = grad_fn
        self.grad = 0.0

    # Computes the gradient of the current variable based on the function input value, wich is the multiplication of the gradient computed with grad_fn and the previous value
    def backprop(self, bp):
        self.grad += bp
        for input, grad in self.grad_fn():
            input.backprop(grad * bp)

    # Runs backpropagation on f, recursivly finding the gradient for each variable composing f
    def backward(self):
        self.backprop(1.0)

    # Define operations:
    # For f(a, b), return a new variable containing the result of the function and the derivates function df/da, df/db
    def __add__(self: 'Var', other: 'Var') -> 'Var':
        return Var(self.v + other.v, lambda: [(self, 1.0), (other, 1.0)])

    def __mul__(self: 'Var', other: 'Var') -> 'Var':
        return Var(self.v * other.v, lambda: [(self, other.v), (other, self.v)])

    def __pow__(self, power):
        assert type(power) in {float, int}, "power must be float or int"
        return Var(self.v ** power, lambda: [(self, power * self.v ** (power - 1))])

    def __neg__(self: 'Var') -> 'Var':
        return Var(-1.0) * self

    def __sub__(self: 'Var', other: 'Var') -> 'Var':
        return self + (-other)

    def __truediv__(self: 'Var', other: 'Var') -> 'Var':
        return self * other ** -1

    def __repr__(self):
        return "Var(v=%.4f, grad=%.4f)" % (self.v, self.grad)

    def relu(self):
        return Var(self.v if self.v > 0.0 else 0.0, lambda: [(self, 1.0 if self.v > 0.0 else 0.0)])

    def identity(self):
        return Var(self.v, lambda: [(self, 1.0)])

    def tanh(self):
        return Var((1.0 - exp(-2.0 * self.v)) / (1.0 + exp(-2.0 * self.v)), lambda: [(self, 1.0 - pow((1.0 - exp(-2.0 * self.v)) / (1.0 + exp(-2.0 * self.v)), 2))])

    def sigmoid(self):
        return Var(1.0/(1.0 + exp(-self.v)), lambda: [(self, (1.0/(1.0 + exp(-self.v))) * (1.0 - 1.0/(1.0 + exp(-self.v))))])

    def exp(self):
        return Var(exp(self.v), lambda: [(self, exp(self.v))])

    def log(self):
        return Var(log(self.v), lambda: [(self, self.v ** -1)])

import numpy as np
import matplotlib.pyplot as plt
from Var import *
from Utils import *
from NeuralNetwork import *

np.random.seed(42)


def data_generator(noise=0.1, n_samples=300, D1=True):
    """
    Creates an example data set to train on
    """

    # Create covariates and response variable
    if D1:
        X = np.linspace(-3, 3, num=n_samples).reshape(-1, 1)  # 1-D
        np.random.shuffle(X)
        y = np.random.normal(
            (0.5*np.sin(X[:, 0]*3) + X[:, 0]), noise)  # 1-D with trend
    else:
        X = np.random.multivariate_normal(
            np.zeros(3), noise*np.eye(3), size=n_samples)  # 3-D
        np.random.shuffle(X)
        y = np.sin(X[:, 0]) - 5*(X[:, 1]**2) + 0.5*X[:, 2]  # 3-D

    # Stack them together vertically to split data set
    data_set = np.vstack((X.T, y)).T

    train, validation, test = np.split(
        data_set, [int(0.35*n_samples), int(0.7*n_samples)], axis=0)

    # Standardization of the data, remember we do the standardization with the training set mean and standard deviation
    train_mu = np.mean(train, axis=0)
    train_sigma = np.std(train, axis=0)

    train = (train-train_mu)/train_sigma
    validation = (validation-train_mu)/train_sigma
    test = (test-train_mu)/train_sigma

    x_train, x_validation, x_test = train[:,
                                          :-1], validation[:, :-1], test[:, :-1]
    y_train, y_validation, y_test = train[:, -
                                          1], validation[:, -1], test[:, -1]

    return x_train, y_train,  x_validation, y_validation, x_test, y_test


# Generate the actual data
D1 = True
x_train, y_train,  x_validation, y_validation, x_test, y_test = data_generator(
    noise=0.5, D1=D1)

x_train = nparray_to_VarArray(x_train)
y_train = nparray_to_VarArray(y_train)
x_validation = nparray_to_VarArray(x_validation)
y_validation = nparray_to_VarArray(y_validation)
x_test = nparray_to_VarArray(x_test)
y_test = nparray_to_VarArray(y_test)


# Instanciate a Deep Neural Network to create a model from the data
NN = NeuralNetwork([1, 8, 1], [lambda x: x.relu(),
                               lambda x: x.identity()])

# Initialize training hyperparameters
EPOCHS = 100
LEARN_R = 2e-3

train_loss, val_loss = NN.trainModel(
    x_train, y_train, x_validation, y_validation, EPOCHS, LEARN_R)


output_test = NN.forward(x_test)
x_test_np = VarArray_to_nparray(x_test)
x_train_np = VarArray_to_nparray(x_train)
y_train_np = VarArray_to_nparray(y_train)
y_test_np = VarArray_to_nparray(y_test)
if D1:
    plt.scatter(x_train_np, y_train_np, label="train data")
    plt.scatter(x_test_np, VarArray_to_nparray(
        output_test), label="test prediction")
    plt.scatter(x_test_np, y_test_np, label="test data")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
else:
    plt.scatter(x_train_np[:, 1], y_train, label="train data")
    plt.scatter(x_test_np[:, 1], VarArray_to_nparray(
        output_test), label="test data prediction")
    plt.scatter(x_test_np[:, 1], y_test_np, label="test data")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
plt.show()

Loss_test = NN.squared_loss(y_test, NN.forward(x_test))

print("Test loss:  {:4.3f}".format(Loss_test.v))

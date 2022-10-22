from Var import *
import numpy as np


# Convert from nparray to Var
def nparray_to_VarArray(x):
    if x.ndim == 1:
        y = [[Var(float(x[i]))]
             for i in range(x.shape[0])]  # always work with list of list
    else:
        y = [[Var(float(x[i, j])) for j in range(x.shape[1])]
             for i in range(x.shape[0])]
    return np.array(y)


# convert from Var to ndarray
def VarArray_to_nparray(x):
    y = np.zeros((len(x), len(x[0])))
    for i in range(len(x)):
        for j in range(len(x[0])):
            y[i, j] = x[i][j].v
    return y


# For a given list of targets and corresponding prediction, ompute the accuracy
def accuracy_score(targets, predictions):
    assert len(targets) == len(predictions), \
        "There must be the same number of targets and predictions"
    correct = 0
    nbElmt = len(targets)
    for i in range(nbElmt):
        correct += 1 if (targets[i][0].v == predictions[i]) else 0
    return correct/nbElmt


# For x is an array of Var predicted scores
# returns the corresponding softmax probabilities
def softmax(x):
    exp_x = [x_i.exp() for x_i in x]

    return np.divide(exp_x, sum(exp_x, start=Var(0.)))

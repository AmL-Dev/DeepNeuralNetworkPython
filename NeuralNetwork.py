import matplotlib.pyplot as plt
from DenseLayer import *
from Utils import *


class NeuralNetwork:

    def __init__(self, layers_sizes, layers_act_fn, initializer=NormalInitializer()):
        assert len(layers_sizes) == len(layers_act_fn) + 1, \
            "For n layers, there must be n-1 activation functions"
        # Fill the network with dense layer objects
        self.network = []
        for n_in, n_out, act_fn in zip(layers_sizes[:-1], layers_sizes[1:], layers_act_fn):
            self.network.append(DenseLayer(n_in, n_out, act_fn, initializer))

    def __repr__(self):
        return 'NeuralNetwork: (' + str([('Layer ' + str(i) + ' ' + self.network[i].__repr__()) for i in range(len(self.network))]) + ')'

    # For a given input computes the output of the network
    def forward(self, input):

        def forward_single(self, x):
            for layer in self.network:
                x = layer.forward(x)
            return x

        output = [forward_single(self, input[n])
                  for n in range(len(input))]
        return output

    # Squared loss function with y the output of the network and t the corresponding target
    def squared_loss(self, t, y):

        # check that sizes agree
        assert len(t) == len(
            y), "output y and target t must match in dimension"

        def squared_loss_single(t, y):
            Loss = Var(0.0)
            for i in range(len(t)):  # sum over outputs
                Loss += (t[i]-y[i]) ** 2
            return Loss

        Loss = Var(0.0)
        for n in range(len(t)):  # sum over training data
            Loss += squared_loss_single(t[n], y[n])
        return Loss

    # Cross entropy loss function with h the output of the layer and t the corresponding target
    def cross_entropy_loss(self, t, h):
        # check that sizes agree
        assert len(t) == len(
            h), "output h and target t must match in dimension"

        ce = Var(0.)

        for i in range(len(h)):  # sum over outputs
            q = softmax(h[i])
            ce += Var(-1.) * np.dot(q, t[i]).log()
        return ce/Var(float(len(h)))

    # Once having computed gradients for every weights and biases, this
    # function updates these parameters to perform the gradient decent
    # Precondition: compute gradients for every weights and biases
    def update_parameters(self, learning_rate=0.01):
        for layer in range(len(self.network)):
            # Get the parameters of the current layer
            for p in self.network[layer].parameters():
                p.v -= learning_rate*p.grad  # Updates these parameters to perform the gradient decent

    # Reset the gradient values of parameters back to zero, to be ready for the next backprop
    def zero_gradients(self):
        for layer in range(len(self.network)):
            for p in self.network[layer].parameters():
                p.grad = 0.0

    # Train the neural network with the x_train/y_train data and test the results with the x_validation/y_validation data.
    # Iterate for num_epochs time with a learning rate of learn_rate for the back propagation
    def trainModel(self, x_train, y_train, x_validation, y_validation, EPOCHS, LEARN_R):
        train_loss = []
        val_loss = []

        for e in range(EPOCHS):

            # Forward pass and loss computation
            Loss = self.squared_loss(y_train, self.forward(x_train))

            # Backward pass
            Loss.backward()

            # gradient descent update
            self.update_parameters(LEARN_R)
            self.zero_gradients()

            # Training loss
            train_loss.append(Loss.v)

            # Validation
            Loss_validation = self.squared_loss(
                y_validation, self.forward(x_validation))
            val_loss.append(Loss_validation.v)

            if e % 10 == 0:
                print("{:4d}".format(e),
                      "({:5.2f}%)".format(e/EPOCHS*100),
                      "Train loss: {:4.3f} \t Validation loss: {:4.3f}".format(train_loss[-1], val_loss[-1]))

        # Plot the learning evolution
        plt.plot(range(len(train_loss)), train_loss)
        plt.plot(range(len(val_loss)), val_loss)
        plt.legend()
        plt.show()

        return train_loss, val_loss

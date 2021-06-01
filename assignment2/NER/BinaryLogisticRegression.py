from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""


class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.1  # The learning rate.
    CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
    MAX_ITERATIONS = 100  # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1000  # Minibatch size (only for minibatch gradient descent)

    # ----------------------------------------------------------------------

    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)

    # ----------------------------------------------------------------------

    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / (1 + np.exp(-z))

    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """

        # REPLACE THE COMMAND BELOW WITH YOUR CODE
        p1 = self.sigmoid(np.dot(self.theta, np.transpose(self.x[datapoint])))
        if label == 1:
            return p1
        else:
            return 1 - p1

    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """

        # YOUR CODE HERE

    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """

        # YOUR CODE HERE
        error = self.sigmoid(np.dot(self.theta, np.transpose(self.x[minibatch]))) - self.y[minibatch]
        gradient = np.zeros(len(self.gradient))
        for k in range(len(gradient)):
            gradient[k] = np.average(np.transpose(self.x[minibatch, k]) * error)
        return gradient

    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """

        # YOUR CODE HERE
        return self.x[datapoint] * (self.conditional_prob(1, datapoint) - self.y[datapoint])

    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        # self.init_plot(self.FEATURES)

        # YOUR CODE HERE
        for i in range(self.MAX_ITERATIONS):
            idx = np.random.choice(range(len(self.x)), 1)[0]
            self.gradient = self.compute_gradient(idx)
            self.theta -= self.LEARNING_RATE * self.gradient
            if (i + 1) % 100 == 0:
                print(self.gradient)
                # self.update_plot(np.sum(np.square(self.gradient)))

    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        # self.init_plot(self.FEATURES)

        # YOUR CODE HERE
        i = 0
        self.gradient = np.ones(len(self.theta)) * 2 * self.CONVERGENCE_MARGIN
        while np.max(np.abs(self.gradient)) > self.CONVERGENCE_MARGIN:
            idx = random.sample(range(len(self.x)), self.MINIBATCH_SIZE)
            self.gradient = self.compute_gradient_minibatch(idx)
            self.theta -= self.LEARNING_RATE * self.gradient
            if i % 100 == 0:
                print(self.gradient)
                # self.update_plot(np.sum(np.square(self.gradient)))
            i += 1

    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        # self.init_plot(self.FEATURES)

        # YOUR CODE HERE
        n = len(self.theta)
        self.gradient = np.ones(n) * 2 * self.CONVERGENCE_MARGIN
        i = 0
        while np.max(np.abs(self.gradient)) > self.CONVERGENCE_MARGIN:
            error = self.sigmoid(np.dot(self.theta, np.transpose(self.x))) - self.y
            for k in range(n):
                # print(np.dot(self.theta, np.transpose(self.x)))
                self.gradient[k] = np.average(np.transpose(self.x[:, k]) * error)

            for k in range(n):
                self.theta[k] -= self.LEARNING_RATE * self.gradient[k]

            if i % 100 == 0:
                print(self.gradient)
                # self.update_plot(np.sum(np.square(self.gradient)))
            i += 1

    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:');

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        # wrong_data = []
        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1
        print("accuracy =", (confusion[0][0] + confusion[1][1]) / np.sum(confusion))
        print("precision no name =", confusion[0][0] / (confusion[0][0] + confusion[0][1]))
        print("recall no name =", confusion[0][0] / (confusion[0][0] + confusion[1][0]))
        print("precision name =", confusion[1][1] / (confusion[1][1] + confusion[1][0]))
        print("recall name =", confusion[1][1] / (confusion[1][1] + confusion[0][1]))
        #     if predicted != self.y[d]:
        #         wrong_data.append(d)
        # print(wrong_data)

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))

    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))

    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)

    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines = []

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5,
                                            markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [1, 1], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0],
        [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [1, 0],
        [1, 0], [0, 0], [1, 1], [0, 0], [1, 0], [0, 0]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()


if __name__ == '__main__':
    main()

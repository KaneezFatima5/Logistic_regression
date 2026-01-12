import time

import numpy as np


def gradient_descent(x, y, lr, w, b, loss_bound):
    """Gradient descent function.

    :param x: features matrix
    :param y: class matrix with one-hot encoding
    :param lr: learning rate
    :param w: weight matrix, each column corresponds to a class
    :param b: bias array, each "column" corresponds to a class
    :param loss_bound: maximum allowable loss
    :return: updated weight matrix and bias array
    """
    num_samples = len(x[0])
    loss = 1
    while (
        loss > loss_bound
    ):  # May need to add option to quit after some number of iterations.
        z = calc_logit(w, x, b)
        p = calc_probabilities(z)
        w = update_weight(w, p, lr, x, y, num_samples)
        b = update_bias(b, p, lr, y, num_samples)
        loss = calc_loss(y, p, num_samples)
    return w, b


def update_weight(w, p, lr, x, y, num_samples):
    """Update the weight in gradient descent"""
    wg = calc_weight_gradient(x, y, p, num_samples)
    return w - lr * wg


def calc_weight_gradient(x, y, p, num_samples):
    """Calculate the weight gradient(s)"""
    return np.matmul(x.T, (p - y)) / num_samples


def update_bias(b, p, lr, y, num_samples):
    """Update the bias in gradient descent"""
    bg = calc_bias_gradient(y, p, num_samples)
    return b - lr * bg


def calc_bias_gradient(y, p, num_samples):
    """Calculate the bias gradient(s)"""
    return np.sum(p - y, axis=0) / num_samples


def calc_logit(x, w, b):
    """Calculate the logit(s) for the sample(s)"""
    return np.matmul(x, w) + b


def calc_probabilities(z):
    """Uses softmax function to calculate probabilities"""
    epz = np.exp(z)  # matrix of e to the power of z_ij
    epz_sum = np.sum(epz, axis=1)[:, np.newaxis]  # sum row-wise and turn into a column
    return epz / epz_sum


def calc_loss(y, p, num_samples):
    """Calculate the loss of the prediction"""
    return -np.sum((y * np.log(p + 1e-10))) / num_samples

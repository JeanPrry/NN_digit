import numpy as np
import pandas as pd


def init_params():

    """
    w1 = np.random.rand(10, 784) - 0.5  # 784 entrées
    b1 = np.random.rand(10, 1) - 0.5    # 10 hiddens neurons / layer1 = 784 --> 10
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5    # layer2 = 10 --> 10
    w3 = np.random.rand(10, 10) - 0.5   # layer3 = 10 --> 10
    b3 = np.random.rand(10, 1) - 0.5    # 10 sorties [0,1,2,..,9]
    """

    weights_and_biases = np.load("./NN_weights_and_biases.npz")
    w1 = weights_and_biases['arr_0']
    b1 = weights_and_biases['arr_1']
    w2 = weights_and_biases['arr_2']
    b2 = weights_and_biases['arr_3']
    w3 = weights_and_biases['arr_4']
    b3 = weights_and_biases['arr_5']
    return w1, b1, w2, b2, w3, b3


def ReLU(z):
    return np.maximum(z, 0)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def forward_prop(w1, b1, w2, b2, w3, b3, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3


def ReLU_deriv(z):
    return z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(z1, a1, z2, a2, z3, a3, w1, w2, w3, X, Y):
    _, m = X.shape

    one_hot_Y = one_hot(Y)
    dz3 = a3 - one_hot_Y
    dw3 = 1 / m * dz3.dot(a2.T)
    db3 = 1 / m * np.sum(dz3)
    dz2 = w3.T.dot(dz3) * ReLU_deriv(z2)
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * ReLU_deriv(z1)
    dw1 = 1 / m * dz1.dot(X.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2, dw3, db3


def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1    
    w2 = w2 - alpha * dw2  
    b2 = b2 - alpha * db2
    w3 = w3 - alpha * dw3
    b3 = b3 - alpha * db3
    return w1, b1, w2, b2, w3, b3


def get_predictions(a3):
    return np.argmax(a3, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    w1, b1, w2, b2, w3, b3 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2, z3, a3 = forward_prop(w1, b1, w2, b2, w3, b3, X)
        dw1, db1, dw2, db2, dw3, db3 = backward_prop(z1, a1, z2, a2, z3, a3, w1, w2, w3, X, Y)
        w1, b1, w2, b2, w3, b3 = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(a3)
            print(get_accuracy(predictions, Y))
    return w1, b1, w2, b2, w3, b3


from math import e

def sigmoid(x):
    return 1 / (1 + (e ** (-x)))

def sigmoid_derivative(x):
    return x * (1 - x)
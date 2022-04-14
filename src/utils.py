import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    z = x - x.max(axis=1)
    num = np.exp(z)
    den = num.sum(axis=1, keepdims=True)
    return num/den
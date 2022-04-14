import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    z = x - x.max(axis=1)
    num = np.exp(z)
    den = num.sum(axis=1, keepdims=True)
    return num/den

def cross_entropy(p_gt, p_pred, reduction="mean"):

    assert reduction in ["mean", "sum"], \
        'Reduction can be either mean or sum'
    assert p_gt.shape == p_pred.shape, \
        "Distributions are not the same, one-hot encoding is expected"

    if reduction == "mean":
        return -(p_gt*np.log(p_pred)).mean()
    else:
        return -(p_gt*np.log(p_pred)).sum()
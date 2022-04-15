from tqdm.notebook import tqdm
from .utils import *
from .dbn import DBN
import numpy as np
import math

class DNN(DBN):

    def __init__(self, channels=[], n_classes=2):
        super().__init__(channels)
        self.head_W = np.random.normal(0, 0.01, (self.ch[-1], n_classes))
        self.head_b = np.zeros((1, n_classes))

    def forward(self, data, get_inter=False):
        if get_inter:
            out = {
                -1:data
            }
        else:
            out = {}

        input = data
        for j in range(self.n_ch-1):
            input = sigmoid(input@self.W[j] + self.b[j])
            if get_inter:
                out[j] = input

        
        output = softmax(input@self.head_W + self.head_b)
        
        if get_inter:
            output = input@self.head_W + self.head_b
            out[self.n_ch-1] = output
            output = softmax(output)
            out["out"] = output
            
            return output, out
        else:
            return softmax(input@self.head_W + self.head_b), out

    def backprop(self, X, y, n_epoches=10, lr=0.1, batch_size=64, shuffle=True):
        losses = []
        size = X.shape[0]
        idxes = np.arange(size)
        if shuffle:
            np.random.shuffle(idxes)

        for _ in (pbar := tqdm(range(n_epoches))):
            err = 0.
            n_elem = 0
            for i in range(math.ceil(size/batch_size)):
                X_batch = X[idxes[i*batch_size:(i+1)*batch_size]]
                y_batch = y[idxes[i*batch_size:(i+1)*batch_size]]
                self.backprop_batch(X_batch, y_batch, lr)

                err += self.eval(X_batch, y_batch)
                n_elem += X_batch.shape[0]

            losses.append(err/n_elem)
            pbar.set_description("{:.5f}".format(losses[-1]))
                    
        return losses

    def backprop_batch(self, X, y, lr):
        _, activations = self.forward(X, get_inter=True)
        grad = activations["out"] - y
        dW = (activations[self.n_ch - 2][:, :, None]@grad[:, None, :]).mean(0)
        db = (grad).mean(0, keepdims=True)
        grad = grad@self.head_W.T
        self.head_W -= lr*dW
        self.head_b -= lr*db

        for j in range(self.n_ch - 2, -1, -1):
            grad = grad*(activations[j]*(1 - activations[j]))
            dW = (activations[j-1][:, :, None]@grad[:, None, :]).mean(0)
            db = (grad).mean(0, keepdims=True)
            grad = grad@self.W[j].T
            self.W[j] -= lr*dW
            self.b[j] -= lr*db

    def eval(self, X, y):
        y_pred, _ = self.forward(X)
        return cross_entropy(y, y_pred, "sum")
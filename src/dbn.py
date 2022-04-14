from tqdm.notebook import tqdm
from .rbm import RBM
from .utils import *
import numpy as np
import math

class DBN():

    def __init__(self, channels=[]):
        assert len(channels), "No layers have been passed"

        self.n_ch = len(channels)
        self.ch = channels
        self.W = [
                    np.random.normal(0, 0.01, (self.ch[i], self.ch[i+1])) 
                        for i in range(self.n_ch - 1)
                    ]
        self.b = [
                    np.zeros((1, n)) 
                        for n in self.ch
                    ]

    def train_DBN(self, data, n_epoches=10, lr=0.1, batch_size=64, shuffle=True):
        losses = []
        size = data.shape[0]
        idxes = np.arange(size)
        if shuffle:
            np.random.shuffle(idxes)
        for _ in (pbar := tqdm(range(n_epoches))):
            err = 0.
            n_elem = 0
            for i in range(math.ceil(size/batch_size)):
                batch = data[idxes[i*batch_size:(i+1)*batch_size]]
                input = batch
                for j in range(self.n_ch-1):
                    self.W[j], self.b[j], self.b[j+1], input = self.train_batch(input, lr, self.W[j], self.b[j], self.b[j+1])

                err += self.evaluate(batch)
                n_elem += batch.shape[0]
            losses.append(err/n_elem)
            pbar.set_description("{:.3f}".format(losses[-1]))
                    
        return losses
    
    def train_batch(self, v_0, lr, W, b_0, b_1):
        p_h_0 = sigmoid(v_0@W + b_1)
        sample_h_0 = np.random.uniform(0, 1, p_h_0.shape)
        h_0 = (sample_h_0 < p_h_0).astype(int)
        
        p_v_1 = sigmoid(h_0@W.T + b_0)
        sample_v_1 = np.random.uniform(0, 1, p_v_1.shape)
        v_1 = (sample_v_1 < p_v_1).astype(int)
        p_h_1 = sigmoid(v_1@W + b_1)
        
        dW = v_0[:, :, np.newaxis]*p_h_0[:, np.newaxis, :] - v_1[:, :, np.newaxis]*p_h_1[:, np.newaxis]
        dW = dW.mean(axis=0)
        
        db_0 = (v_0 - v_1).mean(0, keepdims=True)
        db_1 = (p_h_0 - p_h_1).mean(0, keepdims=True)

        W += lr*dW
        b_0 += lr*db_0
        b_1 += lr*db_1

        p_h_0 = sigmoid(v_0@W + b_1)
        sample_h_0 = np.random.uniform(0, 1, p_h_0.shape)
        h_0 = (sample_h_0 < p_h_0).astype(int)

        return W, b_0, b_1, h_0
    
    def evaluate(self, v_0):        
        return np.linalg.norm(
                    v_0 - self.sample_from_data(v_0),
                    ord=2,
                    axis=1,
                ).mean()
    

    def sample_from_data(self, data):
        v = data
        for j in range(self.n_ch-1):
            p_h = sigmoid(v@self.W[j] + self.b[j+1])
            sample = np.random.uniform(0, 1, p_h.shape)
            v = (sample < p_h).astype(int)

        h = v
        for j in range(self.n_ch-2, -1, -1):
            p_v = sigmoid(h@self.W[j].T + self.b[j])
            sample = np.random.uniform(0, 1, p_v.shape)
            h = (sample < p_v).astype(int)

        return h
        

    def sample_Gibbs(self, n_iters=10, n_images=1, noise = None):
        if noise is not None:
            input = noise
        else:
            input = np.random.randint(0, 2, size=(n_images, self.W[-1].shape[0]))

        for _ in range(n_iters):
            p_v = sigmoid(input@self.W[-1].T + self.b[-2])
            sample_v = np.random.uniform(0, 1, p_v.shape)
            v = (sample_v < p_v).astype(int)
            
            p_h =  sigmoid(v@self.W[-1] + self.b[-1])
            sample_h = np.random.uniform(0, 1, p_h.shape)
            input = (sample_h < p_h).astype(int)

        for j in range(self.n_ch - 2, -1, -1):
            p_v = sigmoid(input@self.W[j].T + self.b[j])
            sample_v = np.random.uniform(0, 1, p_v.shape)
            input = (sample_v < p_v).astype(int)

        return input

class DBN_RBM():

    def __init__(self, layers=[]):

        assert len(layers), "No layers have been passed"
        for layer in layers:
            assert isinstance(layer, RBM), "The layers have to be of Restriced Boltzman Machines class"
        self.layers = layers

    def train_DBN(self, data, n_epoches=10, lr=0.1, batch_size=64, shuffle=True):
        losses = []
        input = data
        for j, layer in enumerate(self.layers):
            losses.append(layer.train_RBM(
                input,
                n_epoches = n_epoches,
                lr = lr,
                batch_size = batch_size,
                shuffle = shuffle
            ))
            p_h = layer.forward(input)
            sample_h = np.random.uniform(0, 1, p_h.shape)
            input = (sample_h < p_h).astype(int)

        return losses
    
    def sample_Gibbs(self, n_iters=10, n_images=1, noise = None):
        input = self.layers[-1].sample_Gibbs(
                                    n_iters, 
                                    n_images,
                                    noise
                                    )

        for layer in self.layers[:-1][::-1]:
            h = input
            for _ in range(n_iters):
                p_v = layer.backward(h)
                sample_v = np.random.uniform(0, 1, p_v.shape)
                v = (sample_v < p_v).astype(int)
                
                p_h = layer.forward(v)
                sample_h = np.random.uniform(0, 1, p_h.shape)
                h = (sample_h < p_h).astype(int)
                
            p_v = layer.backward(h)
            sample_v = np.random.uniform(0, 1, p_v.shape)
            v = (sample_v < p_v).astype(int)
            input = v
            
        return input


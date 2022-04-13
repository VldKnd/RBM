from tqdm.notebook import tqdm
from .rbm import RBM
import numpy as np
import math

class DBN():
    def __init__(self, layers=[]):

        assert len(layers), "No layers have been passed"
        for layer in layers:
            assert isinstance(layer, RBM), "The layers have to be of Restriced Boltzman Machines class"

    
        pass
        
    def init_RMB(self, in_channels, out_channels):
        self.W = np.random.normal(0, 0.01, (in_channels, out_channels))
        self.a = np.zeros((1, out_channels))
        self.b = np.zeros((1, in_channels))
    
    def forward(self, data):
        output = sigmoid(data@self.W + self.a)
        return output
        
    def backward(self, data):
        output = sigmoid(data@self.W.T + self.b)
        return output
        
    def train_RBM(self, data, n_epoches=10, lr=0.1, batch_size=64, shuffle=True):
        losses = []
        n = data.shape[0]
        idxes = np.arange(n)
        if shuffle:
            np.random.shuffle(idxes)
        for e in (pbar := tqdm(range(n_epoches))):
            err = 0.
            n_elem = 0
            for i in range(math.ceil(n/batch_size)):
                batch = data[idxes[i*batch_size:(i+1)*batch_size]]
                self.train_batch(batch, lr)
                err += self.evaluate(batch)
                n_elem += batch.shape[0]
            losses.append(err/n_elem)
            pbar.set_description("{}".format(losses[-1]))
            
        return losses
    
    def evaluate(self, v_0):
        return np.linalg.norm(
                    v_0 - self.sample(v_0),
                    ord=2,
                    axis=1,
                ).mean()
    
    def train_batch(self, v_0, lr):
        p_h_0 = self.forward(v_0)
        sample_h_0 = np.random.uniform(0, 1, p_h_0.shape)
        h_0 = (sample_h_0 < p_h_0).astype(int)
        
        p_v_1 = self.backward(h_0)
        sample_v_1 = np.random.uniform(0, 1, p_v_1.shape)
        v_1 = (sample_v_1 < p_v_1).astype(int)
        p_h_1 = self.forward(v_1)
        
        dW = v_0[:, :, np.newaxis]*p_h_0[:, np.newaxis, :] - v_1[:, :, np.newaxis]*p_h_1[:, np.newaxis]
        dW = dW.mean(axis=0)
        
        da = (p_h_0 - p_h_1).mean(0, keepdims=True)
        db = (v_0 - v_1).mean(0, keepdims=True)

        self.W += lr*dW
        self.a += lr*da
        self.b += lr*db
    
        return self
    
    def sample(self, data):
        p_h_0 = self.forward(data)
        sample = np.random.uniform(0, 1, p_h_0.shape)
        p_v_1 = self.backward(
                (sample < p_h_0).astype(int)
            )
        
        sample = np.random.uniform(0, 1, p_v_1.shape)
        return (sample < p_v_1).astype(int)

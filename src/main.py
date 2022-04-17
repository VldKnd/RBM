from .utils import *
from .data import *
from .dnn import *
from .dbn import *
from .rbm import *

def lire_alpha_digit(idx = "0"):
    _, _, key_to_set = get_binaryalphadigts()
    return key_to_set[idx]

def init_RBM(in_channels, out_channels):
    return RBM(in_channels, out_channels)

def entree_sortie_RBM(rbm, data):
    return rbm.forward(data)

def sortie_entree_RBM(rbm, data):
    return rbm.backward(data)

def train_RBM(rbm, data, n_epoches=10, lr=0.1, batch_size=64):
    losses = rbm.train_RBM(data, n_epoches, lr, batch_size)
    return rbm, losses

def generer_image_RBM(rbm, n_iters=10, n_images=1):
    return rbm.sample_Gibbs(n_iters, n_images)

def init_DNN(channels=[], n_classes=2):
    return DNN(channels, n_classes)

def pretrain_DNN(dnn, data, n_epoches=10, lr=0.1, batch_size=64, ):
    losses = dnn.train_DBN(data, n_epoches, lr, batch_size)
    return dnn, losses

def generer_image_DBN(dbn, n_iters=10, n_images=1):
    return dbn.sample_Gibbs(n_iters, n_images)

def calcul_softmax(x):
    return softmax(x)

def entree_sortie_reseau(dnn, data):
    return dnn.forward(data, get_inter=True)

def retropropagation(dnn, X, y, n_epoches=10, lr=0.1, batch_size=64):
    losses = dnn.backprop(X, y, n_epoches, lr, batch_size)
    return dnn, losses

def test_DNN(dnn, X, y):
    return dnn.eval(X, y)
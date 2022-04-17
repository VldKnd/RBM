import scipy.io
import numpy as np

def get_mnist():
    path = "./data/"
    file = "mnist_all.mat"
    mat = scipy.io.loadmat(path+file)

    X_train = np.concatenate([mat["train"+str(l)] for l in range(10)])
    y_train = np.concatenate([np.ones((mat["train"+str(l)].shape[0], 1))*l for l in range(10)]).astype(int)
    X_test = np.concatenate([mat["test"+str(l)] for l in range(10)])
    y_test = np.concatenate([np.ones((mat["test"+str(l)].shape[0], 1))*l for l in range(10)]).astype(int)

    y_one_hot_train = np.zeros((y_train.shape[0], y_train.max()+1))
    y_one_hot_train[np.arange(y_train.shape[0]), y_train[:, 0]] = 1

    y_one_hot_test = np.zeros((y_test.shape[0], y_test.max()+1))
    y_one_hot_test[np.arange(y_test.shape[0]), y_test[:, 0]] = 1

    return (X_train > 127).astype(float), (X_test > 127).astype(float), y_one_hot_train, y_one_hot_test

def get_binaryalphadigts():
    path = "./data/"
    file = "binaryalphadigs.mat"
    mat = scipy.io.loadmat(path+file)

    raw_X = mat["dat"]
    keys = mat["classlabels"]
    raw_shape = raw_X[0][0].shape

    keys_to_idx = dict(zip(range(36), [str(i[0]) for i in keys[0]]))
    idx_to_keys = dict(zip([str(i[0]) for i in keys[0]], range(36)))

    key_to_set = dict(zip(
            [str(i[0]) for i in keys[0]],
            [np.concatenate([*map(lambda i: i.reshape(1, -1), x)]) for x in raw_X]
        ))
    return keys_to_idx, idx_to_keys, key_to_set
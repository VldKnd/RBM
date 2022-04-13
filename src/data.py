import scipy.io
import numpy as np

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
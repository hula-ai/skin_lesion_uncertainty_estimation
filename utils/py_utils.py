import numpy as np
import h5py


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def one_hot(y):
    y_ohe = np.zeros((y.size, int(y.max() + 1)))
    y_ohe[np.arange(y.size), y.astype(int)] = 1
    return y_ohe


def load_data(path):
    h5f = h5py.File(path, 'r')
    x_train = h5f['X_train'][:]
    y_train = one_hot(h5f['y_train'][:])
    x_valid = h5f['X_test'][:]
    y_valid = one_hot(h5f['y_test'][:])
    h5f.close()

    return x_train, y_train, x_valid, y_valid



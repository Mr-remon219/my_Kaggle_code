import numpy as np

def step_function(x):
    return np.array(x > 0, dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(z):
    if z.ndim == 2:
        z = z.T
        z = z - np.max(z, axis=0, keepdims=True)
        s = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
        return s.T
    C = np.max(z)
    return np.exp(z - C) / np.sum(np.exp(z - C))


def cross_entropy_error_batch(y_hat, y):
    if y_hat.ndim == 1:
        y_hat = y_hat.reshape(1, y_hat.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(y * np.log(y_hat + delta)) / batch_size

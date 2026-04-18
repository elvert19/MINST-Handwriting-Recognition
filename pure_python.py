import numpy as np
import time


def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float64)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)



def cce_loss(y_pred, y_true):
    eps = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class NumpyNet:
    def __init__(self, lr=0.001):
        self.params = {
            "W1": np.random.randn(784, 128) * np.sqrt(2.0 / 784),
            "b1": np.zeros((1, 128)),
            "W2": np.random.randn(128, 64)  * np.sqrt(2.0 / 128),
            "b2": np.zeros((1, 64)),
            "W3": np.random.randn(64, 10)   * np.sqrt(2.0 / 64),
            "b3": np.zeros((1, 10)),
        }
        self.opt = Adam(lr=lr)
        self.cache = {}

    def forward(self, X):
        Z1 = X @ self.params["W1"] + self.params["b1"]
        A1 = relu(Z1)
        Z2 = A1 @ self.params["W2"] + self.params["b2"]
        A2 = relu(Z2)
        Z3 = A2 @ self.params["W3"] + self.params["b3"]
        A3 = softmax(Z3)
        self.cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "A3": A3}
        return A3

    def backward(self, y_true):
        n = y_true.shape[0]
        c = self.cache

        dZ3 = (c["A3"] - y_true) / n
        dW3 = c["A2"].T @ dZ3
        db3 = dZ3.sum(axis=0, keepdims=True)

        dA2 = dZ3 @ self.params["W3"].T
        dZ2 = dA2 * relu_grad(c["Z2"])
        dW2 = c["A1"].T @ dZ2
        db2 = dZ2.sum(axis=0, keepdims=True)

        dA1 = dZ2 @ self.params["W2"].T
        dZ1 = dA1 * relu_grad(c["Z1"])
        dW1 = c["X"].T @ dZ1
        db1 = dZ1.sum(axis=0, keepdims=True)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        self.opt.update(self.params, grads)

    def predict(self, X):
        return self.forward(X)



def train(X_train, y_train, epochs=20, batch_size=64, lr=0.001, callback=None):
    model = NumpyNet(lr=lr)
    n = X_train.shape[0]
    loss_history = []
    epoch_times = []

    for epoch in range(epochs):
        t0 = time.time()
        idx = np.random.permutation(n)
        X_s, y_s = X_train[idx], y_train[idx]
        epoch_loss = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            Xb = X_s[start:start + batch_size]
            yb = y_s[start:start + batch_size]
            out = model.forward(Xb)
            epoch_loss += cce_loss(out, yb)
            model.backward(yb)
            batches += 1

        avg_loss = epoch_loss / batches
        epoch_time = time.time() - t0
        loss_history.append(avg_loss)
        epoch_times.append(epoch_time)

        if callback:
            callback(epoch, avg_loss, epoch_time)

    return model, loss_history, epoch_times


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    predicted = np.argmax(preds, axis=1)
    return np.mean(predicted == y_test) * 100
__author__ = "Li Tao, ltipchrome@gmail.com"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def get_sample(ps):
    """
    :ps:
    :return:
    """
    coin_idx = random.choice(range(len(ps)))
    x = [1 if _ == coin_idx else 0 for _ in range(len(ps))]
    y = 1 if random.random() <= ps[coin_idx] else 0
    return x, y


def test_get_sample():
    for i in range(10):
        ps = [.01, .99, .8]
        print(get_sample(ps))


class FTLR:
    def __init__(self, l1=1, l2=1, alpha=1., beta=1.):
        self.l1 = l1
        self.l2 = l2
        self.alpha = alpha
        self.beta = beta
        self.fitted_ = False
        self.z = 0
        self.n = 0

    def fit(self, x, y):
        x = np.asarray(x)
        n_features = len(x)
        if not self.fitted_:
            self.coef_ = np.random.randn(n_features)

        p = self.predict_proba(x)
        grad = (p - y)*x
        g2 = grad**2
        sqrt_n = np.sqrt(self.n)
        sigma = 1/self.alpha*(np.sqrt(self.n + g2) - np.sqrt(self.n))
        self.z += grad - sigma*self.coef_
        self.n += g2
        self.coef_ = -(self.z - np.sign(self.z)*self.l1)/((self.beta+sqrt_n)/self.alpha + self.l2)
        mask = np.abs(self.z) <= self.l1
        self.coef_[mask] = 0
        self.fitted_ = True
        return self

    def predict_proba(self, x, y=None):
        t = self.coef_.dot(x)
        p = 1/(1+np.exp(-t))
        return p

    def predict(self, x, cutoff=.5):
        p = self.predict_proba(x)
        if p >= cutoff:
            return 1
        return 0


def moving_average(data_set, periods=50):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


if __name__ == "__main__":
    # test_get_sample()

    ftlr = FTLR(alpha=.01, l1=.1, l2=.1)
    ys = []
    ypred = []
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    for i in range(1000):
        x, y = get_sample([.9, 0.1])
        ys.append(y)
        if ftlr.fitted_:
            yhat = ftlr.predict(x)
            ypred.append(yhat)
        ftlr.fit(x, y)
    rs = np.asarray(ys[1:]) - np.asarray(ypred)
    plt.plot(moving_average(np.abs(rs)))
    plt.show()


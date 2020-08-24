import numpy as np
import math

class Legendre_polynomial():
    """A class that return as instance legendre polynomial of order *q*"""

    def __init__(self, q):
        self.q = q

    def __call__(self, arr):
        out = np.vectorize(self.formula)
        return out(arr)
    
    def formula(self, x):
        Qf = self.q
        M = Qf // 2 if Qf % 2 == 0 else (Qf - 1) // 2
        return sum((-1)**k * x**(Qf - 2 * k) * self.binom(k) for k in range(M+1))
    
    def binom(self, k):
        Qf = self.q
        return math.factorial(2*Qf - 2*k) / 2**Qf / math.factorial(k) / \
             math.factorial(Qf - k) / math.factorial(Qf - 2 * k)

class Fit_polynomial():
    """A class for fitting polynomial curves through least squares.
    Optional *lamb* parameter for weight decay (l2 metric)"""

    def __init__(self, order, lamb=0):
        self.order = order
        self.lamb = lamb
        
    def build_matrix(self, x):
        x = np.array(x)[np.newaxis].transpose()
        M = np.broadcast_to(x, (x.size, self.order+1))
        return np.power(M, np.arange(self.order+1))

    def fit(self, data):
        x, y = zip(*data)
        X = self.build_matrix(x)
        Z = np.dot(X.T, X) + self.lamb * np.identity(len(X.T))
        w, *_ = np.linalg.lstsq(Z, np.dot(X.T, y), rcond=None)
        self.weights = w
        return self

    def __call__(self, arr):
        return np.dot(self.build_matrix(arr), self.weights)

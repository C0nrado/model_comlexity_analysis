"""
This module contains utilities functions designed for this project.
"""

import math
import numpy as np
from resources.polynomials import Fit_polynomial
# from polynomials import Fit_polynomial

def get_sample(data, size, seed=None):
    """extract a random sample of *size* from *data*"""
    np.random.seed(seed)
    x, y = zip(*data)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    indices = indices[:int(size)]
    return np.array(x)[indices], np.array(y)[indices]

def add_noise(arr, noise_std):
    """add a random gaussian offset to each value of *arr*"""
    np.random.seed()
    return arr + np.random.normal(0, math.sqrt(noise_std), size=arr.shape)

def make_batch(order_list, params, data, rounds, lamb=0, seed=None):
    """make a batch of fitting curves for each model specified in *order_list* on *rounds*"""
    N = params['N']
    noise_lvl = params['noise']
        
    batch = []
    for _ in range(rounds):
        sample_x, sample_y = get_sample(data, N, seed)
        sample_y = add_noise(sample_y, noise_lvl)
        fittings = []
        for order in order_list:
            fn = Fit_polynomial(order).fit(zip(sample_x, sample_y))
            fittings.append(fn)
        batch.append(fittings)
    return zip(*batch)

def compute_batch(batch, xs):
    """returns a matrix whose columns are the evaluation of fitted polynomials"""
    return np.concatenate([(g(xs)).reshape(1,-1) for g in batch], axis=0)

def E_out(batch, data):
    """Take a batch of polynomial curves and compute generalization error as in
    E[E_out] = E[var(x) + bias(x)]"""
    xs, target = zip(*data)
    values = compute_batch(batch, xs)
    g_bar = np.mean(values, axis=0)
    g_var = np.var(values, axis=0)
    return  g_var.mean() + np.mean((g_bar - np.array(target))**2)

def compute_error(batch, data):
    """This function compute point-wise the error between each fitted
    polynomial and the target provided in *data*"""
    xs, target = zip(*data)
    values = compute_batch(batch, xs)
    return np.mean((np.r_[target] - values)**2, axis=1)
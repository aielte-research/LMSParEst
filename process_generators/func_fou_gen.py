import random
import numpy as np
import cmath
import torch
from scipy.fftpack import fft


def fbm_increments(n, hurst):
    r = np.zeros(n+1)
    for i in range(n+1):
        if i == 0:
            r[0] = 1
        else:
            r[i] = 1 / 2 * ((i + 1)**(2*hurst) - 2*i **
                            (2*hurst) + (i - 1)**(2*hurst))
    r = np.concatenate([r, r[::-1][1:len(r)-1]])
    lmbd = np.real(fft(r) / (2*n))
    sqrt = [cmath.sqrt(x) for x in lmbd]
    W = fft(sqrt * (np.random.normal(size=2*n) +
                    np.random.normal(size=2*n) * complex(0, 1)))



    W = n**(-hurst) * np.real(W[1:(n )])

    # rescale the for the final T

    return W

    # W1 = n**(-hurst) * \
    #     np.cumsum([0, *np.real(W[1:(n)])])

    # W2 = n**(-hurst) * \
    #     np.cumsum([0, *np.real(W[(n+1):])])

    # # rescale the for the final T
    # W1 = (T ** hurst) * W1
    # W2 = (T ** hurst) * W2
    # return W1, W2

def time_grid_setter(n):
        delta = 1/(n-1)
        return [j*delta for j in range(0, n)]

def old_integrand_setter(n, sigma, alpha, time_grid):
    disc_integrands = np.zeros((n-1, n-1))

    for i in range(n-1):
        for j in range(i+1):
            disc_integrands[i][j] = sigma*np.exp(-alpha*(time_grid[i]-time_grid[j]))

    return disc_integrands

def integrand_setter(sigma, alpha, time_grid):
        # it has to be stored in a mtx in this case, since we aim at simulating according to the non-conditional joint distribution
        # this mtx will be multiplied by the increment vector

    x1 = np.array(time_grid[:-1])
    x1 = np.expand_dims(x1, 1)
    x1 = np.exp(-x1)

    x2 = np.array(time_grid[:-1])
    x2 = np.expand_dims(x2, 1)
    x2 = np.exp(x2)

    mtx = x1 @ x2.transpose()
    mtx = np.tril(mtx)

    disc_integrands = sigma * (mtx ** alpha)

    #disc_integrands = sigma * x1 @ x2.transpose()
    disc_integrands = np.tril(disc_integrands)

    return disc_integrands


def fou(n, hurst, alpha, sigma, initial_value):
    # only equi case

    fbm_incs = fbm_increments(n,hurst)

    time_grid = time_grid_setter(n)

    weights = integrand_setter(sigma,alpha,time_grid)

    #print('weights', weights.shape)
    #print('fbm_incs', fbm_incs.shape)
    mtx = weights @ fbm_incs
    return np.array([initial_value, *[np.exp(-alpha*time_grid[i]) * initial_value +
                             mtx[i-1] for i in range(1, len(time_grid))]])


def gen(n = 200, dt = 1, hurst = 0.5, alpha = 0.5, lambd = 1, sigma = 1, gamma = 0, mu = 0):
    #if gamma_varsigma < 0:
        #varsigma = 1
        #gamma = 1 + gamma_varsigma
    #else:
        #gamma = 1
        #varsigma = 1 - gamma_varsigma


    path = fou(n = n, initial_value = gamma, hurst = hurst,
                  alpha = alpha * (n - 1) * dt, sigma = sigma * ((n - 1) * dt) ** hurst) + mu

    return path



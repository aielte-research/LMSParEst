import warnings
import numpy as np
import cmath
from scipy.fftpack import fft
from fbm import FBM


def base_gen(hurst=0.7, n=2000, method="fft", T = 1):

    """
        One dimensional fraction Brownian motion generator object through different methods.
        The output is a realization of a fBm (W_t) where t in [0, T] using 'n' equally spaced grid points.
        The object inputs:
        @arg hurst: the Hurst exponent of the fBm, [0, 1]
        @arg n: the length of data points
        @arg method: define the generator process to get the fBm path, default 'Cholesky'
        @arg T: the interval endpoint of 't', default [0, T=1]
    """
    d = locals()

    if method == 'fft':
        """
            Input:
                @param H: Hurst exponent in [0, 1]
                @param n: number of grid points
                @param T: the endpoint of the time interval, default T=1
            Output:
                W_t the fBm path for t on [0, T]
            Example:
                fbm = FBMGenerators(
                    hurst = 0.7, n=10000, method='fft', T=1)
                path = fbm.fbm()
            Reference:
                Kroese, D. P., & Botev, Z. I. (2015). Spatial Process Simulation.
                In Stochastic Geometry, Spatial Statistics and Random Fields(pp. 369-404)
                Springer International Publishing, DOI: 10.1007/978-3-319-10064-7_12
                https://sci-hub.se/10.1007/978-3-319-10064-7_12
        """
        if hurst < 0 or hurst > 1:
            return warnings.warn('hurst parameter must be between 0 and 1')

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
        W = n**(-hurst) * \
            np.cumsum([0, *np.real(W[1:(n + 1)])])

        # rescale the for the final T
        W = (T ** hurst) * W
        return W

    elif method in ('cholesky', 'hosking', 'daviesharte'):
        d['n'] -= 1
        return FBM(**d).fbm()

    else:
        pass

def gen(hurst = 0.7, n = 2000, method = 'fft', lambd = 1, d=None):
    if d is not None:
        hurst=d+0.5
    m = n - 1
    scale = lambd * (m ** hurst)

    #t = np.linspace(0, n, n + 1)
    #loc_vec_1 = nu_0 * t ** 0
    #loc_vec_1 = nu_1 * t
    #loc_vec_2 = nu_2 * t ** 2

    #return lambd * base_gen(hurst = hurst, n = m, method = method, T = 1)
    return scale * base_gen(hurst = hurst, n = m, method = method, T = 1)
          


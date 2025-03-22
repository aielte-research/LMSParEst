import torch
import numpy as np
import pywt
from scipy.stats import linregress
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.special import psi, zeta

def estimate_hurst(x, wavelet='db3', j1=3, j2=None):
    """
    Estimate the Hurst parameter of a time series using the wavelet-based estimator 
    (first estimator as described in the linked paper).

    Parameters:
      x : 1D numpy array
          The input time series (e.g., a realization of FBM).
      wavelet : str, optional
          Name of the wavelet to use (default is 'db3').
      j1 : int, optional
          Lower bound (finest scale) for regression (default is 3).
      j2 : int, optional
          Upper bound (coarsest scale) for regression; if None, use maximum level available.

    Returns:
      H_est : float
          The estimated Hurst parameter.
      slope : float
          Estimated regression slope (which equals 2H+1).
      intercept : float
          Regression intercept.
      j_values : ndarray
          The octaves (scales) used in regression.
      y1_values : ndarray
          The bias-corrected log-energy values at each octave.
      weights : ndarray
          The weights (reciprocals of the variances) used in regression.
    """
    # Compute the discrete wavelet transform.
    # pywt.wavedec returns [cA_n, cD_n, cD_(n-1), ..., cD_1] where cD_1 is the finest detail.
    coeffs = pywt.wavedec(x, wavelet)
    # Number of available detail levels (octaves)
    J = len(coeffs) - 1  
    if j2 is None:
        j2 = J
    if j1 < 1 or j1 > J:
        raise ValueError("j1 must be between 1 and the number of detail levels available (got j1={}).".format(j1))
    if j2 < j1 or j2 > J:
        raise ValueError("j2 must be between j1 and the number of detail levels available (got j2={}).".format(j2))

    j_values = []
    y1_values = []
    weights = []
    # Loop over octaves j = j1, j1+1, ... , j2.
    # Note: In pywt, the detail coefficients for scale j (with j=1 as the finest scale) are given by coeffs[-j].
    for j in range(j1, j2 + 1):
        d = coeffs[-j]
        n_j = len(d)
        # S(j): empirical second moment (energy) at scale j
        S_j = np.mean(d ** 2)
        # Compute g(j) = ψ(n_j/2)/ln2 - log₂(n_j/2)
        g_j = psi(n_j / 2.0) / np.log(2) - np.log(n_j / 2.0) / np.log(2)
        # Bias-corrected log-energy:
        y1 = np.log2(S_j) - g_j
        # Variance of log₂ S(j): V₍y₁₎(j) = ζ(2, n_j/2) / (ln2)²
        var_y1 = zeta(2, n_j / 2.0) / (np.log(2) ** 2)
        weight = 1.0 / var_y1

        j_values.append(j)
        y1_values.append(y1)
        weights.append(weight)

    j_values = np.array(j_values)
    y1_values = np.array(y1_values)
    weights = np.array(weights)

    # Weighted linear regression: model y1(j) = slope * j + intercept, with weights w = 1/var_y1.
    S0 = np.sum(weights)
    S1 = np.sum(weights * j_values)
    S2 = np.sum(weights * j_values ** 2)
    S_y = np.sum(weights * y1_values)
    S_jy = np.sum(weights * j_values * y1_values)

    # Calculate regression slope and intercept
    slope = (S0 * S_jy - S1 * S_y) / (S0 * S2 - S1 ** 2)
    intercept = (S_y - slope * S1) / S0

    # The regression model implies slope = 2H + 1, so:
    H_est = (slope - 1) / 2.0

    return H_est #, slope, intercept, j_values, y1_values, weights

class Model():
    def __init__(self, params, state_dict=None):
        self.fc = torch.nn.Linear(10, 10)
        self.baseline = True
        self.diff = params.get('diff', True)
        self.num_cores = params.get('num_cores', 16)
        self.shift = params.get('shift', 0)

    def to(self, d):
        return self

    def cuda(self):
        return self

    def parameters(self):
        return self.fc.parameters()

    def train(self, b):
        pass

    def state_dict(self):
        return self.fc.state_dict()

    def __call__(self, x):
        x = np.asarray(x.cpu())
        if not self.diff:
            x=[np.cumsum(s) for s in x]
        if self.num_cores>1:
            with Pool(self.num_cores) as p:
                est = p.map(lambda s: estimate_hurst(s), x)
        else:
            est = [estimate_hurst(s) for s in x]
        est = [s + self.shift for s in est]
        est = torch.FloatTensor(est)
        return torch.unsqueeze(est, 1)

# Example usage:
if __name__ == '__main__':
    from whittlehurst import fbm

    # Original Hurst value to test with
    Hs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for H in Hs:

        # Generate an fBm realization
        seq = fbm(H=H, n=10000)

        # Calculate the increments, because the estimator works with the fGn spectrum
        #seq = np.diff(seq)

        # Estimate the Hurst exponent
        H_est = estimate_hurst(seq)

        print(H, H_est)

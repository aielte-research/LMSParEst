import torch
import numpy as np
import pywt
from scipy.stats import linregress
from pathos.multiprocessing import ProcessingPool as Pool

def wavelet_hurst_estimator(x, wavelet='db2', max_level=None):
    """
    Estimate the Hurst exponent H of a time series x using the Abry-Veitch wavelet method.
    
    Parameters:
        x: array-like, the time series (e.g. an fGn sample or an fBm sample's increments)
        wavelet: string or pywt.Wavelet, the wavelet to use (default is 'db2')
        max_level: int, maximum decomposition level (if None, it's computed based on length)
    
    Returns:
        H: Estimated Hurst exponent.
        slope: The regression slope from the log-scale diagram.
        intercept: The regression intercept.
        r_value: Correlation coefficient from the regression.
    """
    # Compute the discrete wavelet transform (DWT)
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=max_level)
    
    # Skip the approximation coefficients (coeffs[0]) and focus on detail coefficients.
    detail_coeffs = coeffs[1:]
    
    # Calculate variance of detail coefficients at each scale
    variances = np.array([np.var(detail) for detail in detail_coeffs])
    
    # The scale index j: for DWT, detail coefficients at level j correspond to scale 2^j.
    # Here we assume j = 1, 2, ..., number of detail levels.
    j = np.arange(1, len(variances) + 1)
    
    # Perform linear regression on log2(variance) vs. scale index j.
    log_variances = np.log2(variances)
    slope, intercept, r_value, p_value, std_err = linregress(j, log_variances)
    
    # Theoretically: log2(variance) = log2 C - j*(2H+1)
    # Hence, slope = -(2H+1) => H = (-slope - 1) / 2
    H = (-slope - 1) / 2
    
    return H

class Model():
    def __init__(self, params, state_dict):
        self.fc = torch.nn.Linear(10, 10)
        self.baseline = True
        self.diff = params.get('diff', True)
        self.method = params.get('method', "slope")
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
        try:
            x = np.asarray(x.cpu())
        except:
            x = np.asarray(x)
        if not self.diff:
            x=[np.cumsum(s) for s in x]
        if self.num_cores>1:
            with Pool(self.num_cores) as p:
                if self.method=="slope":
                    est = p.map(wavelet_hurst_estimator, x)
                else:
                    raise
        else:
            if self.method=="slope":
                est = [wavelet_hurst_estimator(s) for s in x]
            else:
                raise
            
        est = [s + self.shift for s in est]
        est = torch.FloatTensor(est)
        return torch.unsqueeze(est, 1)
    
# Example usage:
if __name__ == '__main__':
    from whittlehurst import fbm
    import time
    
    total=0
    
    # Original Hurst value to test with
    Hs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for H in Hs:

        # Generate an fBm realization
        seq = fbm(H=H, n=12800)
        
        start = time.time()
        # Estimate the Hurst exponent
        H_est = wavelet_hurst_estimator(seq)
        total += time.time() - start

        print(H, H_est)
        
    print("Time (s):",total)
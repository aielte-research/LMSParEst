import torch
import numpy as np
from scipy.stats import linregress
from pathos.multiprocessing import ProcessingPool as Pool

def Lm_empirical(x, m=1):
    N = len(x)
    ret = 0
    for k in range(N - m):
        ret += x[k] * x[k + m]
    return ret / (N - m)

def log_regress(x, max_lag=5):
    y = []
    lags = np.arange(1, max_lag + 1)
    for m in lags:
        Lm = Lm_empirical(x, m)
        # Use absolute value to avoid issues with negative autocovariances
        y.append(np.log(np.abs(Lm)))
    slope, intercept, r, p, se = linregress(np.log(lags), y)
    return slope, intercept

def estimate_H_slope(x, max_lag=5):
    slope, _ = log_regress(x, max_lag)
    # slope is approximately 2H - 2, so:
    return slope / 2 + 1

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
        if self.diff:
            x = x[:, :-1] - x[:, 1:]
        if self.num_cores>1:
            with Pool(self.num_cores) as p:
                if self.method=="slope":
                    est = p.map(estimate_H_slope, x)
                else:
                    raise
        else:
            if self.method=="slope":
                est = [estimate_H_slope(s) for s in x]
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
        
        # Calculate the increments
        seq = np.diff(seq)
        
        start = time.time()
        # Estimate the Hurst exponent
        H_est = estimate_H_slope(seq)
        total += time.time() - start

        print(H, H_est)
        
    print("Time (s):",total)
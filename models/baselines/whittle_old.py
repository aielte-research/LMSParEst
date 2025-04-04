import torch
import numpy as np
import scipy.optimize as so
from math import gamma
from pathos.multiprocessing import ProcessingPool as Pool

#source: https://github.com/JFBazille/ICode/blob/master/ICode/estimators/whittle.py

def whittle_fgn(data):
    try:
        return whittle(data, spec_name="fgn")
    except:
        return 1

def whittle_arfima(data):
    return whittle(data, spec_name="arfima")

def whittle(data, spec_name="fgn"):
    """This function compute the Hurst exponent of a signal using
    a maximum of likelihood on periodogram
    """
    nbpoints = len(data)
    nhalfm = int((nbpoints - 1) / 2)
    tmp = np.abs(np.fft.fft(data))
    gammahat = np.exp(2 * np.log(tmp[1:nhalfm + 1])) / (2 * np.pi * nbpoints)
    func = lambda Hurst: whittlefunc(Hurst, gammahat, nbpoints, spec_name)
    return so.fminbound(func, 0, 1)

def whittlefunc(hurst, gammahat, nbpoints, spec_name="fgn"):
    """This is the Whittle function
    """
    if spec_name=="fgn":
        gammatheo = fspec_fgn(hurst, nbpoints)
    elif spec_name=="arfima":
        gammatheo = fspec_arfima(hurst, nbpoints)
    else:
        raise
    qml = gammahat / gammatheo
    return 2 * (2 * np.pi / nbpoints) * np.sum(qml)

def fspec_fgn(hest, nbpoints):
    """This is the spectral density of a fGN of Hurst exponent hest
    """
    hhest = - ((2 * hest) + 1)
    const = np.sin(np.pi * hest) * gamma(- hhest) / np.pi
    nhalfm = int((nbpoints - 1) / 2)
    dpl = 2 * np.pi * np.arange(1, nhalfm + 1) / nbpoints
    fspec = np.ones(nhalfm)
    for i in np.arange(0, nhalfm):
        dpfi = np.arange(0, 200)
        dpfi = 2 * np.pi * dpfi
        fgi = (np.abs(dpl[i] + dpfi)) ** hhest
        fhi = (np.abs(dpl[i] - dpfi)) ** hhest
        dpfi = fgi + fhi
        dpfi[0] = dpfi[0] / 2
        dpfi = (1 - np.cos(dpl[i])) * const * dpfi
        fspec[i] = np.sum(dpfi)
    fspec = fspec / np.exp(2 * np.sum(np.log(fspec)) / nbpoints)
    return fspec

def fspec_arfima(hest, nbpoints):
    """This is the spectral density of an ARFIMA of Hurst exponent hest
    """
    d=hest-0.5
    nhalfm = int((nbpoints - 1) / 2)
    dpl = 2 * np.pi * np.arange(1, nhalfm + 1) / nbpoints
    fspec = np.ones(nhalfm)
    for i in np.arange(0, nhalfm):
        fspec[i] = 2*np.pi * np.abs(2*np.sin(dpl[i]/2))**(-2*d)
    #fspec = fspec / np.exp(2 * np.sum(np.log(fspec)) / nbpoints)
    return fspec

class Model():
    def __init__(self, params, state_dict):
        self.fc = torch.nn.Linear(10, 10)
        self.baseline = True
        self.diff = params.get('diff', True)
        self.num_cores = params.get('num_cores', 16)
        self.shift = params.get('shift', 0)
        self.spec_name = params.get('spec_name', "fgn")

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
                if self.spec_name=="fgn":
                    est = p.map(whittle_fgn, x)
                elif self.spec_name=="arfima":
                    est = p.map(whittle_arfima, x)
                else:
                    raise
        else:
            est = [whittle(s, spec_name=self.spec_name) for s in x]
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

        # Calculate the increments, because the estimator works with the fGn spectrum
        seq = np.diff(seq)
        
        start = time.time()
        # Estimate the Hurst exponent
        H_est = whittle_fgn(seq)
        total += time.time() - start

        print(H, H_est)
        
    print("Time (s):",total)
import torch
import numpy as np
from scipy.special import comb, gamma
from scipy.signal import lfilter
from pathos.multiprocessing import ProcessingPool as Pool


class FouEstimator():
    """
    ---Description---
    Params estimatoion of an 1D fOU process what is based on the discretized
    ergodic estimator what was presented and proofed by David Nualart.
    https://arxiv.org/abs/1703.09372

    ---Input---
    @arg series_length:
    @arg hurst:
    @arg sigma:
    @arg alpha:
    @arg initial_value: default=0.0
    @arg time_scale: default=1.0
    @arg filterType: Daubechies or Classical (second order)

    I. Daubechies. Ten Lectures on Wavelets. SIAM, 1992.
    https://jqichina.files.wordpress.com/2012/02/ten-lectures-of-waveletsefbc88e5b08fe6b3a2e58d81e8aeb2efbc891.pdf

    ---Output---
    hurst: Hurst exponent of the 1D fOU process
    sigma: diffusion coefficient
    fou_param: drift coefficient
    """

    def __init__(self, estimated_param, filterType="Daubechies", dt = 0.01, rng=None):
        self.rng = rng 

        self.estimated_param = estimated_param

        self.filter = filterType
        self.order = 2
        self.dt = dt

    def hurst_estimator(self):
        V1, V2 = self.linear_filter()

        estimated_hurst = 1/2 * np.log2(V2/V1)

        self.hurst = estimated_hurst

    def sigma_estimator(self):
        hurst = self.hurst
        V1 = self.V1
        a = self.a
        L = self.L

        delta = 1/self.series_length
        matrix = np.zeros((L, L))
        matrix_t = np.zeros((L, L))

        for i in range(L):
            matrix[i] = np.arange(L)
            matrix_t[i] = i

        const_filter = np.sum(np.tensordot(a, np.transpose(a), axes=0)
                              * np.abs(matrix_t - matrix)**(2*hurst))
        sigma = (-2 * (V1 / (self.series_length-L) /
                       (const_filter*delta**(2*hurst))))**(1/2)

        self.sigma = sigma

    def alpha_estimator(self):
        numerator = 2 * np.mean(self.path ** 2)
        denominator = (self.sigma**2)*gamma(2*self.hurst + 1)
        self.alpha = (numerator/denominator) ** (-1/2/self.hurst)
        self.alpha /= self.dt * (self.series_length - 1)
        
    def __call__(self, x):

        res = {'hurst': [], 'sigma': [], 'alpha': []}
        for seq in x:
            seq = seq.detach().cpu().numpy()
            self.path = seq
            self.series_length = len(x)
            self.hurst_estimator()
            self.sigma_estimator()
            self.alpha_estimator()
            
            for k in res.keys():
                res[k] += [getattr(self, k)]
                
        estimates = res[self.estimated_param]
        if self.rng is not None:
            estimates = np.clip(estimates, self.rng["a"], self.rng["b"])
        estimates=np.nan_to_num(estimates)
        out = torch.FloatTensor(estimates).unsqueeze(1)
        return out


    def linear_filter(self):

        if self.filter == "Daubechies":
            if self.order == 2:
                a = 1/np.sqrt(2) * np.array([0.482962913144534, -0.836516303737808,
                                             0.224143868042013, 0.12940952255126])
            else:
                a = 1/np.sqrt(2) * np.array([0.482962913144534, -0.836516303737808,
                                             0.224143868042013, 0.12940952255126])
                self.order = 2
        elif self.filter == "Classical":
            "Not working yet."
            mesh = range(0, self.order)
            a = (-1) ** (1 - mesh)/2 ** self.order * comb(self.order, mesh)
        else:
            a = 1/np.sqrt(2) * np.array([0.482962913144534, -0.836516303737808,
                                         0.224143868042013, 0.12940952255126])
            self.order = 2

        L = len(a)
        a_2 = 1/np.sqrt(2) * np.array([0, 0.482962913144534, 0, -0.836516303737808,
                                       0, 0.224143868042013, 0,  0.12940952255126])

        V1 = np.sum(lfilter(b=a, a=1, x=self.path)**2)
        V2 = np.sum(lfilter(b=a_2, a=1, x=self.path)**2)

        self.V1 = V1
        self.V2 = V2
        self.a = a
        self.L = L

        return V1, V2

class Model():
    def __init__(self, params, state_dict):
        self.fc = torch.nn.Linear(10, 10)
        self.baseline = True
        rng = params.get('range', None) 
        dt = params.get('dt', 1)
        estimated_param = params.get('estimated_param', "hurst")
        
        estimator = FouEstimator(estimated_param = estimated_param, dt=dt, rng=rng)
        self.estimator = lambda x: estimator(x)

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
        x = x.cpu()
        return self.estimator(x)
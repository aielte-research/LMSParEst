import torch
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

def variogram(path: list[float], p: float = 1) -> np.float64:
    """
        Default estimator is the 1st order variogram (madogram).
        For variogram set p=2, for rodogram set p=1/2
    """

    """
        n: sample length
        l: distance between 2 datapoints
        p the order of the variation
        p = 1 madogram
        p = 2 variogram
        p = 1/2 rodogram
        sample: fBm path
    """

    sum1 = np.sum([np.abs(path[i] - path[i-1]) **
                   p for i in range(len(path))])
    sum2 = np.sum([np.abs(path[i] - path[i-2]) **
                   p for i in range(len(path))])

    # V_p(l/n)
    def vp(increments, l):
        v_p = (1 / (2 * (len(path) - l))) * increments

        return v_p

        # D_Vp
    D = 2 - 1 / p * \
        ((np.log(vp(sum2, 2)) - np.log(vp(sum1, 1))) / np.log(2))

    return D

class Model():
    def __init__(self, params, state_dict):
        self.fc = torch.nn.Linear(10, 10)
        self.baseline = True
        if params is not None:
            self.diff = params.get('diff', True)
            self.num_cores = params.get('num_cores', 16)
            self.shift = params.get('shift', 0)
        else:
            self.diff = True
            self.num_cores = 16
            self.shift = 0


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
        if not self.diff:
            x=[np.cumsum(s) for s in x]
        if self.num_cores>1:
            with Pool(self.num_cores) as p:
                est = [2 - fd + self.shift for fd in p.map(variogram, x)]
        else:
            est = [2 - variogram(s) + self.shift for s in x]
        est = torch.FloatTensor(est)
        return torch.unsqueeze(est, 1)
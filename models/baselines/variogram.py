import torch
import numpy as np
from whittlehurst import variogram
from pathos.multiprocessing import ProcessingPool as Pool

class Model():
    def __init__(self, params, state_dict=None):
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
        try:
            x = np.asarray(x.cpu())
        except:
            x = np.asarray(x)
        if not self.diff:
            x=[np.cumsum(s) for s in x]
        if self.num_cores>1:
            with Pool(self.num_cores) as p:
                est = [H + self.shift for H in p.map(variogram, x)]
        else:
            est = [variogram(s) + self.shift for s in x]
        est = torch.FloatTensor(est)
        return torch.unsqueeze(est, 1)
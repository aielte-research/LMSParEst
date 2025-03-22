import torch
from hurst import compute_Hc
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

def compute_Hc_walk(s):
    try:
        return compute_Hc(s, kind='random_walk', simplified=False)[0]
    except:
        return 1

def compute_Hc_change(s):
    try:
        return compute_Hc(s, kind='change', simplified=False)[0]
    except:
        return 1


class Model():
    def __init__(self, params, state_dict=None):
        self.fc = torch.nn.Linear(10, 10)
        self.baseline = True
        self.diff = params.get('diff', True)
        self.shift = params.get('shift', 0)
        self.num_cores = params.get('num_cores', 16)

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

        if self.num_cores>1:
            with Pool(self.num_cores) as p:
                if self.diff:
                    est = [H+self.shift for H in p.map(compute_Hc_walk, x)]
                else:
                    est = [H+self.shift for H in p.map(compute_Hc_change, x)]
        else:
            if self.diff:
                est = [compute_Hc_walk(s)+self.shift for s in x]
            else:
                est = [compute_Hc_change(s)+self.shift for s in x]         
        est = torch.FloatTensor(est)
        return torch.unsqueeze(est, 1)
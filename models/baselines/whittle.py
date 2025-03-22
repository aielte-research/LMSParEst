import torch
import numpy as np
from whittlehurst import whittle
from pathos.multiprocessing import ProcessingPool as Pool

class Model():
    def __init__(self, params, state_dict=None):
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
        x = np.asarray(x.cpu())
        if self.diff:
            x = x[:, :-1] - x[:, 1:]
        if self.num_cores>1:
            with Pool(self.num_cores) as p:
                est = p.map(lambda x: whittle(x,self.spec_name), x)
        else:
            est = [whittle(s, spectrum=self.spec_name) for s in x]
        est = [s + self.shift for s in est]
        est = torch.FloatTensor(est)
        return torch.unsqueeze(est, 1)
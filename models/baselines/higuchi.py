import torch
from antropy import higuchi_fd
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

class Model():
    def __init__(self, params, state_dict):
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
        x = np.asarray(x.cpu())
        if not self.diff:
            x=[np.cumsum(s) for s in x]
        if self.num_cores>1:
            with Pool(self.num_cores) as p:
                higuchi_est = [2 - fd + self.shift for fd in p.map(higuchi_fd, x)]
        else:
            higuchi_est = [2 - higuchi_fd(s) + self.shift for s in x]
        higuchi_est = torch.FloatTensor(higuchi_est)
        return torch.unsqueeze(higuchi_est, 1)
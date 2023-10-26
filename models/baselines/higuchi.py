import torch
from antropy import higuchi_fd
import numpy as np

class Model():
    def __init__(self, params, state_dict):
        self.fc = torch.nn.Linear(10, 10)
        self.baseline = True
        self.diff = params.get('diff', True)
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
        x = x.cpu()
        if not self.diff:
            x=[np.cumsum(s) for s in x]
        higuchi_est = [2 - higuchi_fd(s) + self.shift for s in x]
        higuchi_est = torch.FloatTensor(higuchi_est)
        return torch.unsqueeze(higuchi_est, 1)
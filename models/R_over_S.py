import torch
from hurst import compute_Hc

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
        if self.diff:
            est = [compute_Hc(s, kind='random_walk', simplified=False)[0]+self.shift for s in x]
        else:
            est = [compute_Hc(s, kind='change', simplified=False)[0]+self.shift for s in x]         
        est = torch.FloatTensor(est)
        return torch.unsqueeze(est, 1)

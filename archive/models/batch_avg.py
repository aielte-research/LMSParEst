import torch
import torch.nn as nn

class BatchAvg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            batch_size = x.size(0)
            data_dim = len(x.size()) - 1
            ones = [1 for _ in range(data_dim)]
            x = torch.mean(x, dim = 0)
            x = x.repeat(batch_size, *ones)
        return x
        

import torch
import torch.nn as nn
from models.base import Hom, MLP, Conv1dRegressor as Base

class DoubleHead(nn.Module):
    def __init__(self, hom_mlp, inhom_mlp):
        super().__init__()
        self.inhom_mlp = MLP(**inhom_mlp)
        self.hom_mlp = MLP(**hom_mlp)

    def forward(self, x):
        scale_infer = self.hom_mlp(x)
        normed_x = x / scale_infer
        param_infer = self.inhom_mlp(normed_x)
        return torch.cat((param_infer, scale_infer), 1)
    
class Model(Base):
    def __init__(self, params, state_dict):
        base_params = {k: v for k, v in params.items() 
                       if k not in ('hom_mlp', 'inhom_mlp')}
                         
        super(Model, self).__init__(base_params)

        self.mlp = DoubleHead(params['hom_mlp'], params['inhom_mlp'])

        if state_dict!=None:
            self.load_state_dict(torch.load(state_dict))


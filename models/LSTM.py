import torch.nn as nn
from models.base import BaseRegressor as Base
from import_tools import get_state_dict, get_state_dict_parallel

class Res_LSTM(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, num_layers=2, bidirectional=False, residual=False):
        super().__init__()

        self.num_layers=num_layers
        self.residual=residual

        if num_layers>0:
            if residual:
                self.lstms = nn.ModuleList([nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional) for i in range(num_layers)])
            else:
                self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
    def forward(self, x):
        if self.num_layers==0:
            return x
        if self.residual:
            for lstm in self.lstms:
                x=x+lstm(x)[0]
            return x
        else:
            return self.lstm(x)[0]

class LSTM(Res_LSTM):
    def __init__(self, params):
        default_params={
            "input_size": 64,
            "hidden_size": 64,
            "num_layers": 2,
            "bidirectional": False,
            "residual": False
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        super().__init__(**params)

        #the lstm weights need to be initialized manually as by default they are suboptimal
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        return super().forward(x.permute(2, 0, 1)).permute(1, 2, 0)
        
    
class Model(Base):
    def __init__(self, params, state_dict):
        super(Model, self).__init__(params)

        assert("lstm" in params)

        self.seq2seq = LSTM(params["lstm"])

        if state_dict!=None:
            try:
                self.load_state_dict(get_state_dict(state_dict))
            except: #if it was accidentally saved as a parallel model, should not happen anymore
                self.load_state_dict(get_state_dict_parallel(state_dict))
                



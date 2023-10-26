import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var

#device = torch.cuda.current_device()

class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)
    
class My_Conv1d(nn.Sequential):
    def __init__(self, conv1d_kwargs, batch_norm=True, dropout=0, spatial_dropout=False):
        seq = [nn.Conv1d(**conv1d_kwargs)]
        if batch_norm:
            seq.append(nn.BatchNorm1d(num_features = conv1d_kwargs["out_channels"]))
        seq.append(nn.ReLU())
        if spatial_dropout:
            seq.append(Permute(perm = (0, 2, 1)))
            seq.append(nn.Dropout2d(p = dropout))
            seq.append(Permute(perm = (0, 2, 1)))
        elif dropout>0:
            seq.append(nn.Dropout(p = dropout))  

        super(My_Conv1d, self).__init__(*seq)
    
class Conv1dRegressor(nn.Module):
    def __init__(self, params):
        super(Conv1dRegressor, self).__init__()
        
        default_params={
            "feature_size":20,
            "window_sizes":[3,5,7,9],
            "dropout": 0,
            "in_channels": 1
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])
            
        self.dropout_rate = params['dropout']
        self.convs = nn.ModuleList([nn.Sequential(My_Conv1d({'in_channels': params["in_channels"],
                                                                 'out_channels': params["feature_size"],
                                                                 'kernel_size': h},
                                                                 batch_norm = False
                                                                 ),
                                                      nn.AdaptiveAvgPool1d(1)
                                                     )
                                         for h in params["window_sizes"]
                                       ])

        self.fc = nn.Sequential(nn.Linear(params["feature_size"]*len(params["window_sizes"]), params["feature_size"]),
                                nn.ReLU(),
                                nn.Linear(params["feature_size"], 1))

    def forward(self, batch):
        output = [conv(batch) for conv in self.convs]
        out = torch.cat(output, dim=1)
        out = out.view(-1, out.size(1)) 
        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.fc(out)
        #out = out.squeeze(dim=1)
        return out
    
class Model(Conv1dRegressor):
    def __init__(self, params, state_dict=None):
        super(Model, self).__init__(params)
        if state_dict!=None:
            self.load_state_dict(torch.load(state_dict))

    def forward(self, batch):
        return super(Model, self).forward(batch.unsqueeze(dim = 1))

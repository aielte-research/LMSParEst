import math
import functools as ft
import torch
from torch import nn
import torch.nn.functional as F

device = torch.cuda.current_device()

# Fc is a more concise way to declare fully connected neural nets with ReLU activation
#e.g. Fc(100, 50, 25, 1) is equivalent to 
#nn.Sequential(
#  nn.Linear(in_features=100, out_features=50),
#  nn.ReLU(),
#  nn.Linear(in_features=50, out_features=25)
#  nn.ReLU()
#  nn.Linear(25, 1))

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

def Fc(neurons, leaky_relu = False, slope = 0.3,  bias = True):

    if leaky_relu:
        re = lambda: nn.LeakyReLU(slope)

    else:
        re = lambda: nn.ReLU()

    layers = ft.reduce(lambda x,y: x + [re(), nn.Linear(x[-1].out_features, y, bias = bias)],
                                        neurons[2:], [nn.Linear(*neurons[0:2], bias = bias)] )
    return nn.Sequential(*layers)

class LSTM_Conv1D_mixed(nn.Module):
    def __init__(self, params):
        super().__init__()

        for k, v in params.items():
            setattr(self, k, v)

        #first a homogeneous layer that will norm our input
        hom_neurons = [int(self.hom_size * x) for x in self.hom_neuron_ratio] + [1]
        self.hom = Fc(hom_neurons, bias = False, leaky_relu = self.leaky_relu,
                      slope = self.slope)
        
        #then a 1D pooling layer to increase the dimension of our input
        self.conv = nn.Conv1d(in_channels = 1,
                out_channels = self.lstm_params['input_size'],
                kernel_size = self.kernel_size)

        #then the lstm module itself
        self.lstm = nn.LSTM(**self.lstm_params)

        #the lstm weights need to be initialized manually as by default they are suboptimal
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        #then an average pooling layer
        self.avg = nn.AvgPool1d(kernel_size = self.pool_size)

        #then we flatten the output 
        self.flat = nn.Flatten()

        #then we need to calculate the input size for the Fc
        if self.use_only_last_hidden:
            s = self.lstm_params['hidden_size'] * self.num_last_hidden

        else:
            s = int((self.input_size - self.kernel_size + 1)/self.pool_size) * self.lstm_params['hidden_size'] 

        #and declare it
        self.conv1D_params["in_channels"] = self.lstm_params['hidden_size']
        self.fc = Conv1dRegressor(self.conv1D_params)

    def forward(self, x):
        
        if self.avg_infer:
            n = x.size(1)

            pool_size = math.ceil(n / self.input_size)
            new_size = pool_size * self.input_size

            part0 = x[:, :(new_size - self.input_size)]
            part1 = x[:, -self.input_size:]

            x = torch.cat((part0, part1), dim = 1)

            x = x.view(x.size(0) * pool_size, self.input_size)

            first_col = torch.unsqueeze(x[:, 0], dim = 1)
            x = x - first_col 
            
            x = self._forward(x)

            x = torch.split(x, pool_size, dim = 0)
            means = [torch.mean(t, dim = 0) for t in x]

            return torch.stack(means, dim = 0)

        else:
            return self._forward(x)

    def _forward(self, x):
        #applying the modules defined in init
        #some transformations are necessary dimensionwise

        if self.simple_norming:
            first_val = torch.unsqueeze(x[:, 1], 1)
            x = x / first_val

        elif self.use_hom_layer: 
            x = x/(self.hom(x[:, :self.hom_size]) + 10**(-10))

        x = torch.unsqueeze(x, -1)

        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = torch.transpose(x, 1, 2)

        if self.use_only_last_hidden:
            x = self.lstm(x)[0][:, -self.num_last_hidden:, :]

        else:
            x = self.lstm(x)[0]
            x = torch.transpose(x, 1, 2)
            x = self.avg(x)
            #x = torch.transpose(x, 1, 2)
            #the last line is only because of backward compability reasons

        #x = self.flat(x)

        return self.fc(x)



class Model(LSTM_Conv1D_mixed):
    def __init__(self, params = {}, state_dict = None):
        default_params={
            'simple_norming': False,
            'use_hom_layer': True,
            'avg_infer': True,
            'use_only_last_hidden': False,
            'num_last_hidden': 50,

            'input_size': 1500,
            'hom_size': 100,
            'hom_neuron_ratio': [1, 1, 0.75, 0.75, 0.5],
            'leaky_relu': False,
            'slope': 0.3,
            'kernel_size': 32,
            'lstm_params': {'input_size': 64, 'hidden_size': 64, 'num_layers': 2},
            'conv1D_params': {'feature_size': 80, 'window_sizes': [3,5,7,11], 'dropout': 0.2},
            'pool_size': 16,
            'fc_neuron_ratio': [1, 0.5]
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        super(Model, self).__init__(params)

        if state_dict!=None:
            self.load_state_dict(torch.load(state_dict))


#before usage we can check if the model works for the desired input size

#lstm = Model(params = {'input_size': 100}).to(device)

#v = torch.randn(4, 322).to(device)

#print(lstm(v))
#print(lstm(4*v))


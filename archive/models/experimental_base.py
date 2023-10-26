import torch
import torch.nn as nn
import torch.nn.functional as f



class Permute(nn.Module):
    def __init__(self, perm):
        super().__init__()
        self.perm = perm
    def forward(self, x):
        return x.permute(*self.perm)

def activation_Function(name="ReLU",params=None):
    #print(name,params)
    if name==None:
        return nn.Identity()
    elif name in ["ELU","Hardshrink","Hardsigmoid","Hardtanh","Hardswish",
                  "LeakyReLU","LogSigmoid","MultiheadAttention","PReLU","ReLU",
                  "ReLU6","RReLU","SELU","CELU","GELU","Sigmoid","SiLU","Mish",
                  "Softplus","Softshrink","Softsign","Tanh","Tanhshrink",
                  "Threshold","GLU", "Identity"]:
        if params==None:
            return getattr(nn, name)()
        return getattr(nn, name)(*params)
    raise ValueError("Unknown activation function!",name)

class MLP(nn.Sequential):
    def __init__(self, channels=[64], batch_norm=False, dropout=0, activation={"name": "ReLU"}, bias = True):
        seq = []
        for i in range(len(channels)-2):
            seq.append(nn.Linear(*channels[i:i+2], bias = bias))
            seq.append(activation_Function(**activation))
            if dropout>0:
                seq.append(nn.Dropout(p = dropout))
            if batch_norm:
                seq.append(nn.BatchNorm1d(num_features = channels[i+1]))
        
        seq.append(nn.Linear(*channels[-2:], bias = bias))

        super().__init__(*seq)


#THREE DIFFERENT KINDS OF HOM LAYERS
#----------------------------------------------------------------------------

class MLConv1d(nn.Sequential):
    def __init__(self, channels, kernel_sizes, strides=1, dilations=1, bias = True, activation = {'name': 'PReLU', 'params': None},
                       dropout=0, spatial_dropout=False, batch_norm=False,
                       adaptive_out_length = 1, leave_last_layer_bare=False):
        in_channels = channels[:-1]
        out_channels = channels[1:]

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes for _ in in_channels]
        if isinstance(strides, int):
            strides = [strides for _ in in_channels]
        if isinstance(dilations, int):
            dilations = [dilations for _ in in_channels]

        seq = []
        for i, o, k, s, d in zip(in_channels, out_channels, kernel_sizes, strides, dilations):
            seq.append(nn.Conv1d(in_channels = i, out_channels = o, kernel_size = k, bias = bias, dilation=d, stride=s))
            seq.append(activation_Function(**activation))
            if dropout>0:
                if spatial_dropout:
                    seq.append(Permute(perm = (0, 2, 1)))
                    seq.append(nn.Dropout2d(p = dropout))
                    seq.append(Permute(perm = (0, 2, 1)))
                else:
                    seq.append(nn.Dropout(p = dropout))
            if batch_norm:
                seq.append(nn.BatchNorm1d(num_features = o))

        if leave_last_layer_bare:
            if batch_norm:
                seq.pop()
            if dropout>0:
                seq.pop()
                if spatial_dropout:
                    seq.pop()
                    seq.pop()
            seq.pop()

        super().__init__(*seq)
    
#MU LAYER
#------------------------------------------------------------------------------

class LinearConv1dRegressor(nn.Module):
    def __init__(self, **params):
        super().__init__()
        new_params = {k: v for k, v in params.items() 
                      if k not in ('activation', 'bias')}
        self.conv1d = MLConv1d(**new_params, activation = {'name': 'Identity'},
                               bias = False)
        self.avg = nn.AdaptiveAvgPool1d(params["adaptive_out_length"])
        self.flatten = nn.Flatten()
        final_dim = params["adaptive_out_length"] * params["channels"][-1]
        self.mlp = nn.Linear(final_dim, 1, bias = False)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1d(x)
        x = self.avg(x)
        x = self.flatten(x)
        out = self.mlp(x)
        return out

class HomRegressor(nn.Module):
    def __init__(self, params):
        super().__init__()

        default_params={
            "diff": False,
            "embedding":{
                "channels": [1, 64],
                "kernel_sizes": 4,
                "dilations": 1, 
                "strides": 1,
                "activation":{
                    "name": "PReLU",
                    "params": None
                },
                "dropout": 0,
                "spatial_dropout": False,
                "batch_norm": False
            },
            "adaptive_out_length": 1,
            "mlp":{
                "channels": [64,32,1],
                "batch_norm": False,
                "dropout": 0,
                "activation":{
                    "name": "LeakyReLU",
                    "params": [0.3]
                }
            }
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        self.diff = params["diff"]
        self.emb = MLConv1d(**params["embedding"])
        self.avg = nn.AdaptiveAvgPool1d(params["adaptive_out_length"])
        self.mlp = MLP(**params["mlp"])

    def forward(self, x):
        if self.diff:
            x = x[:, :-1] - x[:, 1:]

        #print("x",x.size())
        x = x.unsqueeze(dim = 1)
        #print("x = x.unsqueeze(dim = 1)",x.size())
        x = self.emb(x)
        #print("x = self.emb(x)",x.size())
        x = self.avg(x)
        #print("x = self.avg(x)",x.size())
        x = x.view(-1, x.size(1) * x.size(2)) 
        #print("x = x.view(-1, x.size(1))",x.size())
        out = self.mlp(x)
        #print("out = self.mlp(x",out.size())
        #input()
        return out

class MainRegressor(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        default_params={
            "embedding":{
                "channels": [1, 64],
                "kernel_sizes": 4,
                "dilations": 1, 
                "strides": 1,
                "activation":{
                    "name": "PReLU",
                    "params": None
                },
                "dropout": 0,
                "spatial_dropout": False,
                "batch_norm": False
            },
            "adaptive_out_length": 1,
            "seq2seq": None,
            "mlp":{
                "channels": [64,32,1],
                "batch_norm": False,
                "dropout": 0,
                "activation":{
                    "name": "LeakyReLU",
                    "params": [0.3]
                }
            },
            "hom_multiplier": None
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        
        self.emb = MLConv1d(**params["embedding"])

        #seq2seq module, placeholder does nothing
        self.seq2seq = nn.Identity()

        #average the seq2seq output
        self.avg = nn.AdaptiveAvgPool1d(params["adaptive_out_length"])

        #Infer the param from the averages
        self.mlp = MLP(**params["mlp"])


    def forward(self, x):

        #print("x",x.size())
        #print("x = self.hom(x)",x.size())
        x = x.unsqueeze(dim = 1)
        #print("x = x.unsqueeze(dim = 1)",x.size())
        x = self.emb(x)
        #print("x = self.emb(x)",x.size())
        x = self.seq2seq(x)
        #print("x = self.seq2seq(x)",x.size())
        x = self.avg(x)
        #print("x = self.avg(x)",x.size())
        x = x.view(-1, x.size(1) * x.size(2)) 
        #print("x = x.view(-1, x.size(1))",x.size())
        out = self.mlp(x)
        #print("out = self.mlp(x",out.size())
        #input()
        return out
class MuLayer(nn.Module):
    def __init__(self, **params):
        super().__init__()
        lin_conv_params = {k: v for k, v in params.items() if k != 'v_0'}
        self.linear_conv1d_regressor = LinearConv1dRegressor(**lin_conv_params)
        self.v_0 = params['v_0']
        
    def forward(self, x):
        T = torch.linspace(0, 1, x.size(-1)).repeat(x.size(0), 1)
        if self.v_0 == 'const_1/N':
            v_0 = torch.FloatTensor([1/x.size(1)]).repeat(*x.size())
        elif self.v_0 == 'const_1':
            v_0 = torch.FloatTensor([1]).repeat(*x.size())
        elif self.v_0 == 'linear':
            v_0 = torch.linspace(0, 1, x.size(-1)).repeat(x.size(0), 1)
        else:
            raise ValueError('Unknown MuLayer v_0 type!', self.v_0)

        v_0 = v_0.to(x.device)
        numerator = self.linear_conv1d_regressor(x)
        denominator = self.linear_conv1d_regressor(v_0)
        out = x - (numerator / denominator) * v_0
        return out
#------------------------------------------------------------------------------

class BaseRegressor(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        default_params={
            "hom_regressor":
            {
                "embedding":{
                    "channels": [1, 64],
                    "kernel_sizes": 4,
                    "dilations": 1, 
                    "strides": 1,
                    "activation":{
                        "name": "PReLU",
                        "params": None
                    },
                    "dropout": 0,
                    "spatial_dropout": False,
                    "batch_norm": False
                },
                "adaptive_out_length": 1,
                "mlp":{
                    "channels": [64,32,1],
                    "batch_norm": False,
                    "dropout": 0,
                    "activation":{
                        "name": "LeakyReLU",
                        "params": [0.3]
                    }
                }
                },
            "main_regressor":
            {
                "embedding":{
                    "channels": [1, 64],
                    "kernel_sizes": 4,
                    "dilations": 1, 
                    "strides": 1,
                    "activation":{
                        "name": "PReLU",
                        "params": None
                    },
                    "dropout": 0,
                    "spatial_dropout": False,
                    "batch_norm": False
                },
                "adaptive_out_length": 1,
                "mlp":{
                    "channels": [64,32,1],
                    "batch_norm": False,
                    "dropout": 0,
                    "activation":{
                        "name": "LeakyReLU",
                        "params": [0.3]
                    }
                }
                },
            "scale_estimator": False
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        #first a homogeneous layer that will norm our input
        if params["mu_layer"] is not None:
            self.mu_layer = MuLayer(**params['mu_layer'])
        else:
            self.mu_layer = nn.Identity()

        if params["init_hom"] is not None:
            self.init_hom = Hom(**params['init_hom'])
        else:
            self.init_hom = nn.Identity()
        
        #then a 1D pooling/embedding layer to increase the dimension of our input
        self.emb = MLConv1d(**params["embedding"])


        #average the seq2seq output
        self.avg = nn.AdaptiveAvgPool1d(params["adaptive_out_length"])

        if params["vec_hom"] is not None:
            self.vec_hom = Hom(**params['vec_hom'])
        else:
            self.vec_hom = nn.Identity()

        #Infer the param from the averages
        self.mlp = MLP(**params["mlp"])

        self.conv_regressors = []
        for _ in range(complexity_level):
            self.hom_regressors.append(HomRegressor(params["hom_regressor"]))

        self.main_regressors = []
        for _ in range(complexity_level):
            self.main_regressors.append(MainRegressor(params["main_regressor"]))

        if self.scale_estimator:
            self.last_hom_regressor = HomRegressor(params["hom_regressor"])

        else:
            self.last_hom_regressor = nn.Identity()

    def forward(self, x):
        for hr, mr in zip(self.hom_regressors, self.main_regressors):
            x = x / hr(x)
            x = mr(x)
        
        x = x * self.last_hom_regressor(x)
            
class Model(Conv1dRegressor):
    def __init__(self, params, state_dict=None):
        super().__init__(params)
        if state_dict!=None:
            self.load_state_dict(torch.load(state_dict))



import torch
import torch.nn as nn
import torch.nn.functional as f

def get_state_dict(state_dict):
    if torch.cuda.is_available():
        return torch.load(state_dict)
    return torch.load(state_dict,map_location=torch.device('cpu'))

def get_state_dict_parallel(state_dict): #if it was accidentally saved as a parallel model 
    from collections import OrderedDict
    state_dict=get_state_dict(state_dict)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] #remove 'module'
        new_state_dict[name] = v
    return new_state_dict

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


class MLConv1d(nn.Sequential):
    def __init__(self, channels, kernel_sizes, conv_dim = 1, strides=1, dilations=1,
                 paddings = 0, bias = True,
                 activation = {'name': 'PReLU', 'params': None},
                 dropout=0, spatial_dropout=False, batch_norm=False,
                 adaptive_out_length = 1, leave_last_layer_bare=False):

        if conv_dim == 1:
            Conv = nn.Conv1d
            BatchNorm = nn.BatchNorm1d
        elif conv_dim == 2:
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
        else:
            raise ValueError('conv_dim has to be 1 or 2')

        in_channels = channels[:-1]
        out_channels = channels[1:]

        def issimple(param, dtype):
            if conv_dim == 1:
                return isinstance(param, dtype)
            elif conv_dim == 2:
                return isinstance(param, dtype) or isinstance(param[0], dtype)

        if issimple(kernel_sizes, int):
            kernel_sizes = [kernel_sizes for _ in in_channels]
        if issimple(strides, int):
            strides = [strides for _ in in_channels]
        if issimple(dilations, int):
            dilations = [dilations for _ in in_channels]
        if issimple(paddings, int) or issimple(paddings, str):
            paddings = [paddings for _ in in_channels]


        seq = []
        for i, o, k, s, d, p in zip(in_channels, out_channels, kernel_sizes,
                                    strides, dilations, paddings):
            seq.append(Conv(in_channels = i, out_channels = o, kernel_size = k,
                            bias = bias, dilation = d, stride = s, padding = p))
            seq.append(activation_Function(**activation))
            if dropout>0:
                if spatial_dropout:
                    seq.append(Permute(perm = (0, 2, 1)))
                    seq.append(nn.Dropout2d(p = dropout))
                    seq.append(Permute(perm = (0, 2, 1)))
                else:
                    seq.append(nn.Dropout(p = dropout))
            if batch_norm:
                seq.append(BatchNorm(num_features = o))

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

#------------------------------------------------------------------------------

class BaseRegressor(nn.Module):
    def __init__(self, params):
        super().__init__()

        default_params={
            "avg_base": None,
            "diff": False,
            "standardize": False,
            "adaptive_out_length": 1,
            "embedding":{
                "conv_dim": 1,
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
        self.standardize = params["standardize"]


        #then a 1D pooling/embedding layer to increase the dimension of our input
        if params["embedding"] is not None:
            self.emb = MLConv1d(**params["embedding"])
        else:
            self.emb = nn.Identity()

        #seq2seq module, placeholder does nothing
        self.seq2seq = nn.Identity()

        #average the seq2seq output
        
        if params["adaptive_out_length"] is not None:
            self.avg = nn.AdaptiveAvgPool1d(params["adaptive_out_length"])
        else:
            self.avg = nn.Identity()

        self.flatten = nn.Flatten()

        #Infer the param from the averages
        self.mlp = MLP(**params["mlp"])

        self.avg_base = params["avg_base"]

    def forward(self, x):
        if self.diff:
            x = x[:, :-1] - x[:, 1:]

        size_cond = (self.avg_base is not None) and (x.size(1) > self.avg_base)
        if size_cond:
            n_mult = int(x.size(1) / self.avg_base)
            x = x.view(x.size(0) * n_mult, -1)

        if self.standardize:            
            m = x.mean(dim = 1, keepdim = True)
            s = x.std(dim = 1, keepdim = True)
            x = (x - m) / s

        #print("x",x_p.size())
        x = x.unsqueeze(dim = 1)
        #print("x = x.unsqueeze(dim = 1)",x.size())
        x = self.emb(x)
        #print("x = self.emb(x)",x.size())
        x = self.seq2seq(x)
        #print("x = self.seq2seq(x)",x.size())
        x = self.avg(x)
        #print("x = self.avg(x)",x.size())
        #x = x.view(-1, x.size(1) * x.size(2))
        #print("x = x.view(-1, x.size(1))",x.size())
        x = self.flatten(x)
        out = self.mlp(x)
        #print("out = self.mlp(x",out.size())

        if size_cond:
            out = out.view(-1, 2)
            out = out.mean(axis = 1, keepdim = True)

        return out

class Model(BaseRegressor):
    def __init__(self, params, state_dict=None):
        super().__init__(params)
        if state_dict!=None:
            try:
                self.load_state_dict(get_state_dict(state_dict))
            except: #if it was accidentally saved as a parallel model, should not happen anymore
                self.load_state_dict(get_state_dict_parallel(state_dict))
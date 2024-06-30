import torch
import sys
sys.path.append('../')
from models.baselines.R_over_S import Model as R_over_S
from models.baselines.variogram import Model as Variogram
from models.baselines.higuchi import Model as Higuchi
from models.baselines.whittle import Model as Whittle
from models.LSTM import Model as LSTM
from models.base import Model as Conv1d

def to_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var

def load_models(diff=True, cuda=True):
    r_over_s=R_over_S({'diff':diff},None)
    variogram=Variogram({'diff':diff},None)
    higuchi=Higuchi({'diff':diff},None)
    whittle=Whittle({'diff':diff},None)

    model_params={
        "diff": True,
        "standardize": True,
        "invariant_nu_layer": None,
        "additive_nu_layer": None,
        "init_hom": None,
        "embedding": None,
        "vec_hom": None,
        "lstm":{
            "input_size": 1,
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": False,
            "residual": False
        },
        "adaptive_out_length": 1,
        "mlp":{
            "bias": True,
            "channels": (128, 64, 1),
            "batch_norm": False,
            "dropout": 0,
            "activation":{"name": "PReLU"}
        }
    }

    state_dict_path="../model_checkpoints/fBm/fBm_Hurst_LSTM_finetune_until_n-12800.pt"
    lstm = LSTM(model_params, state_dict_path)
    if cuda:
        lstm = to_cuda(lstm)
    lstm.eval()

    model_params={
        "diff": True,
        "standardize": True,
        "invariant_nu_layer": None,
        "additive_nu_layer": None,
        "init_hom": None,
        "rs": None,
        "embedding": {
            "bias": True,
            "channels": (1, 64, 64, 128, 128, 128, 128),
            "kernel_sizes": 4,
            "activation":{"name": "PReLU"}
        },
        "vec_hom": None,
        "avg_base": None,
        "nu_estimator": False,
        "simple_nu_estimator": False,
        "lstm":{
            "input_size": 1,
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": False,
            "residual": False
        },
        "adaptive_out_length": 1,
        "mlp":{
            "bias": True,
            "channels": (128, 64, 1),
            "batch_norm": False,
            "dropout": 0,
            "activation":{"name": "PReLU"}
        }
    }

    state_dict_path="../model_checkpoints/fBm/fBm_Hurst_conv1D_finetune_until_n-12800.pt"
    
    conv1d = Conv1d(model_params, state_dict_path)
    if cuda:
        conv1d = to_cuda(conv1d)
    conv1d.eval()
    
    return r_over_s, variogram, higuchi, whittle, lstm, conv1d
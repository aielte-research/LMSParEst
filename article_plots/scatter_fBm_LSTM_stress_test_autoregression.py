import random
import torch
import numpy as np
from bokeh.palettes import Category10
from tqdm import trange

import sys
sys.path.append('..')
from metrics.plotters import general_plot, scatter_plot
from process_generators.autoregression_gen import gen as autoregression_gen
from models.LSTM import Model as LSTM
from models.baselines.R_over_S import Model as R_over_S
from models.baselines.variogram import Model as Variogram
from models.baselines.higuchi import Model as Higuchi
from models.baselines.whittle import Model as Whittle

diff=True
num_cores=33
r_over_s=R_over_S({'diff':diff, "num_cores": num_cores},None)
variogram=Variogram({'diff':diff, "num_cores": num_cores},None)
higuchi=Higuchi({'diff':diff, "num_cores": num_cores},None)
whittle=Whittle({'diff':diff, "num_cores": num_cores},None)

def to_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var

model_params={
    "diff": False,
    "standardize": False,
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

state_dict_path="../model_checkpoints/ARFIMA/ARFIMA_d_LSTM_finetune_until_n-12800.pt"
lstm_arfima = LSTM(model_params, state_dict_path)
lstm_arfima = to_cuda(lstm_arfima)
lstm_arfima.eval()

model_params={
    "diff": False,
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
lstm_fbm = LSTM(model_params, state_dict_path)
lstm_fbm = to_cuda(lstm_fbm)
lstm_fbm.eval()

model_params={
    "diff": False,
    "standardize": True,
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

state_dict_path="../model_checkpoints/fOU/fOU_Hurst_LSTM_finetune_until_n-12800.pt"
lstm_fou = LSTM(model_params, state_dict_path)
lstm_fou = to_cuda(lstm_fou)
lstm_fou.eval()


scatter_Ys_r_over_s=[]
scatter_Ys_variogram=[]
scatter_Ys_higuchi=[]
scatter_Ys_whittle=[]
scatter_Xs=[]
scatter_Ys_lstm=[]
scatter_Ys_lstm_fou=[]
scatter_Ys_lstm_arfima=[]
for _ in trange(10):
    inputs=[]
    inputs_diff=[]
    for __ in range(200):
        alpha = random.uniform(-1, 1)
        scatter_Xs.append(alpha)

        process = autoregression_gen(3200, alpha = alpha)
        inputs.append(process)
        inputs_diff.append(np.diff(process))

    input=to_cuda(torch.FloatTensor(inputs))
    input_diff=to_cuda(torch.FloatTensor(inputs_diff))

    print("LSTM...")
    scatter_Ys_lstm += [float(val[0]) for val in lstm_fbm(input_diff).detach().cpu()]
    scatter_Ys_lstm_fou += [float(val[0]) for val in lstm_fou(input).detach().cpu()]
    scatter_Ys_lstm_arfima += [float(val[0])+0.5 for val in lstm_arfima(input_diff).detach().cpu()]
    # print("R_over_S...")
    # scatter_Ys_r_over_s += [float(val) if val > -1 else -1 for val in r_over_s(input.cpu())]
    # print("Variogram...")
    # scatter_Ys_variogram += [float(val) if val > -1 else -1 for val in variogram(input.cpu())]
    # print("Higuchi...")
    # scatter_Ys_higuchi += [float(val) if val > -1 else -1 for val in higuchi(input.cpu())]
    # print("Whittle...")
    # scatter_Ys_whittle += [float(val) if val > -1 else -1 for val in whittle(input.cpu())]

# Autoregression fBm

scatter_plot({
    "Xs": [scatter_Xs]*3, 
    "Ys": [scatter_Ys_lstm_fou, scatter_Ys_lstm, scatter_Ys_lstm_arfima],#, scatter_Ys_r_over_s, scatter_Ys_variogram, scatter_Ys_higuchi, scatter_Ys_whittle],
    "xlabel": "Speed of Mean Reversion (α)",
    "ylabel": "Inferred Hurst",
    "title": "",
    "fname": "scatter_fBm_LSTM_stress_test_autoregression",
    "dirname": "./plots",
    "circle_size": 6,
    "x_jitter": 0,
    "opacity": 0,
    "heatmap": False,
    "boundary": {
        "functions": ['x if x<0.55 else None'],
        "dashes": ["solid"],
        "colors": ["black"],
        "legend": [None],
    },
    "line45_color": None,#"black",
    "legend": {
        "labels": ["trained on fOU", "trained on fBm", "trained on ARFIMA"], #+["R/S","variogram","Higuchi","Whittle"],
        "location": "bottom_right"
    },
    "matplotlib": {
        "width": 5,
        "height": 4,
        # "width": 6,
        # "height": 4.5,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})
import random
import torch
import numpy as np
from bokeh.palettes import Category10
from tqdm import tqdm

import sys
sys.path.append('..')
from metrics.plotters import general_plot, scatter_plot
from process_generators.fbm_gen import gen as fbm_gen
from models.LSTM import Model as LSTM
from models.baselines.R_over_S import Model as R_over_S
from models.baselines.variogram import Model as Variogram
from models.baselines.higuchi import Model as Higuchi
from models.baselines.whittle import Model as Whittle

diff=True
num_cores = 42
r_over_s=R_over_S({'diff':diff, "num_cores": num_cores}, None)
variogram=Variogram({'diff':diff, "num_cores": num_cores}, None)
higuchi=Higuchi({'diff':diff, "num_cores": num_cores}, None)
whittle=Whittle({'diff':diff, "num_cores": num_cores}, None)


def to_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var

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

state_dict_path="../model_checkpoints/fBm/fBm_Hurst_LSTM_n-1600.pt"
lstm = to_cuda(LSTM(model_params, state_dict_path))
lstm.eval()

state_dict_path="../model_checkpoints/fBm/fBm_Hurst_LSTM_n-100.pt"
lstm_100 = to_cuda(LSTM(model_params, state_dict_path))
lstm_100.eval()

state_dict_path="../model_checkpoints/fBm/fBm_Hurst_LSTM_n-400.pt"
lstm_400 = to_cuda(LSTM(model_params, state_dict_path))
lstm_400.eval()

state_dict_path="../model_checkpoints/fBm/fBm_Hurst_LSTM_n-6400.pt"
lstm_6400 = to_cuda(LSTM(model_params, state_dict_path))
lstm_6400.eval()

state_dict_path="../model_checkpoints/fBm/fBm_Hurst_LSTM_n-12800.pt"
lstm_12800 = to_cuda(LSTM(model_params, state_dict_path))
lstm_12800.eval()

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

orig=[]
est=[]
est_100=[]
est_400=[]
est_6400=[]
est_12800=[]
est_r_over_s=[]
est_variogram=[]
est_higuchi=[]
est_whittle=[]
for _ in range(10):
    inputs=[]
    for __ in range(100):
        H = random.uniform(0, 1)
        orig.append(H)

        
        process = [x**H for x in np.linspace(0.001, 1.0, num=12800)]
        #process = [x**H for x in range(1,12801)]
        inputs.append(process)

    print("LSTM...")    
    
    input=to_cuda(torch.FloatTensor(inputs))
    est_12800 += [float(val[0]) for val in lstm_12800(input).detach().cpu()]

    input=to_cuda(torch.FloatTensor([inp[:100] for inp in inputs]))
    est_100 += [float(val[0]) for val in lstm_100(input).detach().cpu()]

    input=to_cuda(torch.FloatTensor([inp[:400] for inp in inputs]))
    est_400 += [float(val[0]) for val in lstm_400(input).detach().cpu()]

    input=to_cuda(torch.FloatTensor([inp[:6400] for inp in inputs]))
    est_6400 += [float(val[0]) for val in lstm_6400(input).detach().cpu()]

    input=to_cuda(torch.FloatTensor([inp[:1600] for inp in inputs]))
    est += [float(val[0]) for val in lstm(input).detach().cpu()]

    print("r_over_s...")
    est_r_over_s += [float(val) for val in r_over_s(input.cpu())]
    print("variogram...")
    est_variogram += [float(val) for val in variogram(input.cpu())]
    print("higuchi...")
    est_higuchi += [float(val) for val in higuchi(input.cpu())]
    print("whittle...")
    est_whittle += [float(val) for val in whittle(input.cpu())]

#mse = np.square(np.asarray(orig) - np.asarray(est)).mean()

scatter_plot({
    "Xs": [orig]*9,
    "Ys": [est_r_over_s, est_variogram, est_higuchi, est_whittle, est_100, est_400, est, est_6400, est_12800],
    "xlabel": "H",
    "ylabel": "Inferred Value",
    "title": "",
    "fname": "stress_test_scatter_fBm_Hurst_LSTM_power_",
    "dirname": "./plots",
    "circle_size": 5,
    "x_jitter": 0,
    "opacity": 0,
    "heatmap": False,
    "colors": [Category10[10][1],Category10[10][2],Category10[10][3],Category10[10][4], Category10[10][0], Category10[10][5], Category10[10][6], Category10[10][7], Category10[10][8]],
    "line45_color": "black",
    "legend": {
        "location": "bottom_right",
        "labels": ["R/S","variogram","Higuchi","Whittle","LSTM 100","LSTM 400","LSTM 1600","LSTM 6400","LSTM 12800"],
        "markerscale": 2.
    },
    "matplotlib": {
        "width": 6.7,
        "height": 4,
        # "width": 9,
        # "height": 4.75,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})
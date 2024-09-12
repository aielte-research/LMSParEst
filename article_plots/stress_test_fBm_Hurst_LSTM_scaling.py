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
r_over_s=R_over_S({'diff':diff},None)
variogram=Variogram({'diff':diff, "num_cores": 36},None)
higuchi=Higuchi({'diff':diff},None)
whittle=Whittle({'diff':diff, "num_cores": 36},None)


def to_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var

model_params={
    "diff": True,
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

state_dict_path="../model_checkpoints/fBm_non_std/fBm_Hurst_LSTM_n-1600_standardize-False.pt"
lstm_non_std = to_cuda(LSTM(model_params, state_dict_path))
lstm_non_std.eval()

model_params["standardize"]= True
state_dict_path="../model_checkpoints/fBm/fBm_Hurst_LSTM_finetune_until_n-12800.pt"
lstm = to_cuda(LSTM(model_params, state_dict_path))
lstm.eval()

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
Xs = list(np.exp2(np.power(np.linspace(-2, 2, num=21),3)))
Ys = []
Ys_non_std = []
Ys_r_over_s = []
Ys_variogram = []
Ys_higuchi = []
Ys_whittle = []
scatter_Xs = []

for scale in tqdm(Xs):
    orig=[]
    est=[]
    est_non_std=[]
    est_r_over_s=[]
    est_variogram=[]
    est_higuchi=[]
    est_whittle=[]
    for _ in range(50):
        inputs=[]
        for __ in range(100):
            H = random.uniform(0, 1)
            orig.append(H)

            process = fbm_gen(hurst = H, n = 1600, lambd=scale, scaled=False)
            input = to_cuda(torch.FloatTensor([process]))
            inputs.append(process)

        input=to_cuda(torch.FloatTensor(inputs))
        est += [float(val[0]) for val in lstm(input).detach().cpu()]
        est_non_std += [float(val[0]) for val in lstm_non_std(input).detach().cpu()]

        est_r_over_s += [float(val) for val in r_over_s(input.cpu())]
        est_variogram += [float(val) for val in variogram(input.cpu())]
        est_higuchi += [float(val) for val in higuchi(input.cpu())]
        est_whittle += [float(val) for val in whittle(input.cpu())]

    mse = np.square(np.asarray(orig) - np.asarray(est)).mean()
    Ys.append(mse)

    mse_non_std = np.square(np.asarray(orig) - np.asarray(est_non_std)).mean()
    Ys_non_std.append(mse_non_std)

    Ys_r_over_s.append(np.square(np.asarray(orig) - np.asarray(est_r_over_s)).mean())
    Ys_variogram.append(np.square(np.asarray(orig) - np.asarray(est_variogram)).mean())
    Ys_higuchi.append(np.square(np.asarray(orig) - np.asarray(est_higuchi)).mean())
    Ys_whittle.append(np.square(np.asarray(orig) - np.asarray(est_whittle)).mean())

general_plot({
    "Ys": [Ys_r_over_s,Ys_variogram,Ys_higuchi,Ys_whittle,Ys,Ys_non_std],
    "Xs": Xs,
    "xlabel": "scale",
    "xscale": "log",
    "ylabel": "MSE loss",
    "yscale": "log",
    "title": "",
    "fname": "stress_test_fBm_Hurst_LSTM_scaling",
    "dirname": "./plots",
    "legend": {
        "location": "bottom_left",
        "labels": ["R/S","variogram","Higuchi","Whittle",r"$M_{LSTM}$",r"$M_{LSTM}^*$"]
    },
    "markers": None,
    "colors":  ["red","red","red","red"]+[Category10[10][0]]+[Category10[10][1]],
    "dashes": ["dotted","dashdot","dashed","solid","solid","solid"],
    "line45_color": None,
    "matplotlib": {
        "width": 6,
        "height": 3.5,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})

# scatter_plot({
#     "Xs": [scatter_Xs]*5,
#     "Ys": [scatter_Ys_r_over_s, scatter_Ys_variogram, scatter_Ys_higuchi, scatter_Ys_whittle, scatter_Ys_lstm],
#     "xlabel": "Hurst",
#     "ylabel": "Inferred Value",
#     "title": "",
#     "fname": "stress_test_scatter_fBm_Hurst_LSTM_smoothing",
#     "dirname": "./plots",
#     "circle_size": 5,
#     "x_jitter": 0,
#     "opacity": 0,
#     "heatmap": False,
#     "colors": [Category10[10][1],Category10[10][2],Category10[10][3],Category10[10][4],Category10[10][0]],
#     "line45_color": "black",
#     "legend": {
#         "location": "bottom_right",
#         "labels": ["R/S","variogram","Higuchi","Whittle","LSTM"],
#         "markerscale": 2.
#     },
#     "matplotlib": {
#         "width": 6.7,
#         "height": 4,
#         # "width": 9,
#         # "height": 4.75,
#         "style": "default"
#     },
#     "color_settings":{
#         "bg_transparent": False
#     }
# })
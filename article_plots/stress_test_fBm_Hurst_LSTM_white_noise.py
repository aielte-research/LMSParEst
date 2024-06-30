import random
import torch
import numpy as np
from plotters import general_plot
from bokeh.palettes import Category10
from tqdm import tqdm

import sys
sys.path.append('..')
from process_generators.fbm_gen import gen as fbm_gen
from models.LSTM import Model as LSTM
from models.baselines.R_over_S import Model as R_over_S
from models.baselines.variogram import Model as Variogram
from models.baselines.higuchi import Model as Higuchi
from models.baselines.whittle import Model as Whittle

diff=True
r_over_s=R_over_S({'diff':diff},None)
variogram=Variogram({'diff':diff},None)
higuchi=Higuchi({'diff':diff},None)
whittle=Whittle({'diff':diff},None)


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

state_dict_path="../model_checkpoints/fBm/fBm_Hurst_LSTM_finetune_until_n-12800.pt"
lstm = to_cuda(LSTM(model_params, state_dict_path))
lstm.eval()

Xs = list(np.linspace(0.0, 1, num=20)**2)
Ys = []
Ys_r_over_s = []
Ys_variogram = []
Ys_higuchi = []
Ys_whittle = []
for sigma in tqdm(Xs):
    orig=[]
    est=[]
    est_r_over_s=[]
    est_variogram=[]
    est_higuchi=[]
    est_whittle=[]
    for _ in range(1000):
        H = random.uniform(0, 1)
        orig.append(H)
        process = fbm_gen(hurst = H, n = 256)
        input=to_cuda(torch.FloatTensor([process + np.random.normal(loc=0, scale=sigma, size=len(process))]))
        lstm_estimate = [float(val[0]) for val in lstm(input).detach().cpu()][0]
        est.append(lstm_estimate)
        est_r_over_s.append(float(r_over_s(input.cpu())[0]))
        est_variogram.append(float(variogram(input.cpu())[0]))
        est_higuchi.append(float(higuchi(input.cpu())[0]))
        est_whittle.append(float(whittle(input.cpu())[0]))

    mse = np.square(np.asarray(orig) - np.asarray(est)).mean()
    Ys.append(mse)

    Ys_r_over_s.append(np.square(np.asarray(orig) - np.asarray(est_r_over_s)).mean())
    Ys_variogram.append(np.square(np.asarray(orig) - np.asarray(est_variogram)).mean())
    Ys_higuchi.append(np.square(np.asarray(orig) - np.asarray(est_higuchi)).mean())
    Ys_whittle.append(np.square(np.asarray(orig) - np.asarray(est_whittle)).mean())
    
gp_params={
    "Ys": [Ys,Ys_r_over_s,Ys_variogram,Ys_higuchi,Ys_whittle],
    "Xs": Xs,
    "xlabel": "sigma",
    #"xscale": "log",
    "ylabel": "MSE loss",
    #"yscale": "log",
    "title": "",
    "fname": "stress_test_fBm_Hurst_LSTM_white_noise",
    "dirname": "./plots",
    # "legend": {
    #     "location": "top_right",
    #     "labels": ["fBm Hurst LSTM model n=3200"]
    # },
    "legend": {
        "location": "top_left",
        "labels": ["R/S","variogram","Higuchi","Whittle","LSTM"]
    },
    "markers": None,
    "colors":  ["red","red","red","red"]+[Category10[10][0]],
    "dashes": ["dotted","dashdot","dashed","solid","solid"],
    "line45_color": None,
    "matplotlib": {
        "width": 6,
        "height": 3.5,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
}
general_plot(gp_params)
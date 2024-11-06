import random
import torch
import numpy as np
from bokeh.palettes import Category10
from tqdm import tqdm, trange

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

# def weierstrass(x, a, b=7, iters=100):
#     ws = np.zeros(len(x))
#     for n in range(iters):
#         ws += a**n * np.cos(b**n*np.pi*x)
#     return ws

def weierstrass(x, s, lambd=1.5, iters=200):
    ws = np.zeros(len(x))
    for k in range(1,iters):
        ws += lambd**((s-2)*k) * np.sin(lambd**k*x)
    return ws


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

Xs = []
Ys = []
Ys_r_over_s = []
Ys_variogram = []
Ys_higuchi = []
Ys_whittle = []
ns = [100, 200, 800, 3200, 12800]
b=7

for n in tqdm(ns):
    state_dict_path=f"../model_checkpoints/fBm/fBm_Hurst_LSTM_n-{n}.pt"
    lstm = to_cuda(LSTM(model_params, state_dict_path))
    lstm.eval()

    orig=[]
    est=[]
    est_r_over_s=[]
    est_variogram=[]
    est_higuchi=[]
    est_whittle=[]
    for _ in tqdm(range(5)):
        inputs=[]
        for __ in range(100):

            # a = random.uniform(1/7, 1)
            # orig.append(-np.log(a)/np.log(b))
            s = random.uniform(1, 2)
            orig.append(2-s)
            
            process = weierstrass(np.linspace(0, 1, num=n), s)
            #process = [x**H for x in range(1,12801)]
            inputs.append(process)

        #print("LSTM...")    
        
        input=to_cuda(torch.FloatTensor(inputs))
        est += [float(val[0]) for val in lstm(input).detach().cpu()]

        #print("r_over_s...")
        est_r_over_s += [float(val) for val in r_over_s(input.cpu())]
        #print("variogram...")
        est_variogram += [float(val) for val in variogram(input.cpu())]
        #print("higuchi...")
        est_higuchi += [float(val) for val in higuchi(input.cpu())]
        #print("whittle...")
        est_whittle += [float(val) for val in whittle(input.cpu())]
    
    Xs.append(orig)
    Ys.append(est)
    Ys_r_over_s.append(est_r_over_s)
    Ys_variogram.append(est_variogram)
    Ys_higuchi.append(est_higuchi)
    Ys_whittle.append(est_whittle)

    #mse = np.square(np.asarray(orig) - np.asarray(est)).mean()

    scatter_plot({
        "Xs": [orig]*5,
        "Ys": [est_r_over_s, est_variogram, est_higuchi, est_whittle, est],
        "xlabel": "H",
        "ylabel": "Inferred Value",
        "title": "",
        "fname": f"stress_test_scatter_fBm_Hurst_LSTM_Weierstrass_{n}",
        "dirname": "./Weierstrass",
        "circle_size": 5,
        "x_jitter": 0,
        "opacity": 0,
        "heatmap": False,
        "colors": [Category10[10][1],Category10[10][2],Category10[10][3],Category10[10][4], Category10[10][0]],
        "line45_color": "black",
        "legend": {
            "location": "bottom_right",
            "labels": ["R/S","variogram","Higuchi","Whittle","LSTM"],
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

params={
    "Xs": Xs,
    "Ys": Ys,
    "xlabel": "H",
    "ylabel": "Inferred Value",
    "title": "",
    "fname": f"stress_test_scatter_fBm_Hurst_LSTM_Weierstrass_LSTM",
    "dirname": "./Weierstrass",
    "circle_size": 5,
    "x_jitter": 0,
    "opacity": 0,
    "heatmap": False,
    #"colors": [Category10[10][1],Category10[10][2],Category10[10][3],Category10[10][4], Category10[10][0]],
    "line45_color": "black",
    "legend": {
        "location": "bottom_right",
        "labels": [f"n={n}" for n in ns],
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
}

scatter_plot(params)

params["Ys"] = Ys_r_over_s
params["fname"] = f"stress_test_scatter_fBm_Hurst_LSTM_Weierstrass_r_over_s"
scatter_plot(params)

params["Ys"] = Ys_variogram
params["fname"] = f"stress_test_scatter_fBm_Hurst_LSTM_Weierstrass_variogram"
scatter_plot(params)

params["Ys"] = Ys_higuchi
params["fname"] = f"stress_test_scatter_fBm_Hurst_LSTM_Weierstrass_higuchi"
scatter_plot(params)

params["Ys"] = Ys_whittle
params["fname"] = f"stress_test_scatter_fBm_Hurst_LSTM_Weierstrass_whittle"
scatter_plot(params)
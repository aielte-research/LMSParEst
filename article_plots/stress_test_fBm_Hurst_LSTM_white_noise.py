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

Xs = list(np.linspace(0.0, 1.2, num=31)**2)
print(Xs)
Ys = []
Ys_r_over_s = []
Ys_variogram = []
Ys_higuchi = []
Ys_whittle = []
scatter_Xs = []

for sigma in tqdm(Xs):
    orig=[]
    est=[]
    est_r_over_s=[]
    est_variogram=[]
    est_higuchi=[]
    est_whittle=[]
    for _ in range(5):
        inputs=[]
        for __ in range(5000):
            H = random.uniform(0, 1)
            orig.append(H)
            if sigma==1.0:
                scatter_Xs.append(H)

            process = fbm_gen(hurst = H, n = 1600)
            inputs.append(process + np.random.normal(loc=0, scale=sigma, size=len(process)))

        input=to_cuda(torch.FloatTensor(inputs))

        #print("LSTM...")
        est += [float(val[0]) for val in lstm(input).detach().cpu()]
        #print("R_over_S...")
        est_r_over_s += [float(val) for val in r_over_s(input.cpu())]
        #print("Variogram...")
        est_variogram += [float(val) for val in variogram(input.cpu())]
        #print("Higuchi...")
        est_higuchi += [float(val) for val in higuchi(input.cpu())]
        #print("Whittle...")
        est_whittle += [float(val) for val in whittle(input.cpu())]

        if sigma==1.0:
            scatter_Ys_lstm = est.copy()
            scatter_Ys_r_over_s = est_r_over_s.copy()
            scatter_Ys_variogram = est_variogram.copy()
            scatter_Ys_higuchi = est_higuchi.copy()
            scatter_Ys_whittle = est_whittle.copy()

    mse = np.square(np.asarray(orig) - np.asarray(est)).mean()
    Ys.append(mse)

    Ys_r_over_s.append(np.square(np.asarray(orig) - np.asarray(est_r_over_s)).mean())
    Ys_variogram.append(np.square(np.asarray(orig) - np.asarray(est_variogram)).mean())
    Ys_higuchi.append(np.square(np.asarray(orig) - np.asarray(est_higuchi)).mean())
    Ys_whittle.append(np.square(np.asarray(orig) - np.asarray(est_whittle)).mean())
    
general_plot({
    "Ys": [Ys_r_over_s,Ys_variogram,Ys_higuchi,Ys_whittle, Ys],
    "Xs": Xs,
    "xlabel": "σ of the white noise",
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
        "height": 4.5,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})

scatter_plot({
    "Xs": [scatter_Xs[:2000]]*5,
    "Ys": [scatter_Ys_r_over_s[:2000], scatter_Ys_variogram[:2000], scatter_Ys_higuchi[:2000], scatter_Ys_whittle[:2000], scatter_Ys_lstm[:2000]],
    "xlabel": "Hurst",
    "ylabel": "Inferred value",
    "title": "",
    "fname": "stress_test_scatter_fBm_Hurst_LSTM_white_noise",
    "dirname": "./plots",
    "circle_size": 4,
    "x_jitter": 0,
    "opacity": 0,
    "heatmap": False,
    "colors": [Category10[10][1],Category10[10][2],Category10[10][3],Category10[10][4],Category10[10][0]],
    "line45_color": "black",
    "legend": {
        "location": "top_left",
        "labels": ["R/S","variogram","Higuchi","Whittle","LSTM"],
        "markerscale": 2.
    },
    "matplotlib": {
        "width": 6,
        "height": 4.5,
        # "width": 9,
        # "height": 4.75,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})
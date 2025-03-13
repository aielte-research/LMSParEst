import numpy as  np
import os
import logging
import random
import math
import torch
from bokeh.palettes import Category10
from tqdm import trange

## fBm biases


import sys

sys.path.append('..')
sys.path.append('../..')
from metrics.plotters import general_plot, scatter_plot, scatter_grid_plot
from metrics.deviation import calc_dev
from process_generators.fbm_gen import gen as fbm_gen
from models.LSTM import Model as LSTM
from models.baselines.R_over_S_gpu import Model as R_over_S
from models.baselines.variogram import Model as Variogram
from models.baselines.tst import Model as Tst
from models.baselines.higuchi import Model as Higuchi
from models.baselines.whittle import Model as Whittle
from models.baselines.whittle_gpu import Model as Whittle_gpu
import time

def to_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var

diff = True
num_cores = 32

model_params = {
    "diff": True,
    "standardize": True,
    "invariant_nu_layer": None,
    "additive_nu_layer": None,
    "init_hom": None,
    "embedding": None,
    "vec_hom": None,
    "lstm": {
        "input_size": 1,
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": False,
        "residual": False
    },
    "adaptive_out_length": 1,
    "mlp": {
        "bias": True,
        "channels": (128, 64, 1),
        "batch_norm": False,
        "dropout": 0,
        "activation": {"name": "PReLU"}
    }
}

state_dict_path = "../../model_checkpoints/fBm/fBm_Hurst_LSTM_finetune_until_n-3200.pt"
lstm = torch.nn.DataParallel(to_cuda(LSTM(model_params, state_dict_path)))
lstm.eval()

models = dict(
    R_over_S = torch.nn.DataParallel(to_cuda(R_over_S({'diff': diff, "num_cores": num_cores}))),
    #variogram = Variogram({'diff': diff, "num_cores": num_cores}),
    tst = Tst({'diff': diff, "num_cores": num_cores}),
    #Higuchi = Higuchi({'diff': diff, "num_cores": num_cores}),
    #Whittle = Whittle({'diff': diff, "num_cores": num_cores}),
    WhittleGPU = torch.nn.DataParallel(to_cuda(Whittle_gpu({'diff': diff, "num_cores": num_cores}))),
    LSTM = lstm
)

orig = []
est = {nam: [] for nam in models.keys()}
total = {nam: 0.0 for nam in models.keys()}

pbar=trange(10)
for _ in pbar:
    pbar.set_description("Generating...")
    inputs = []
    for _ in range(500):
        H = random.uniform(0, 1)
        orig.append(H)
        process = fbm_gen(hurst = H, n = 3200)
        inputs.append(np.asarray(process))

    inp_gpu = to_cuda(torch.FloatTensor(np.asarray(inputs)))

    for nam, model in models.items():
        start = time.time()
        pbar.set_description(nam)
        est[nam] += [float(val[0]) for val in model(inp_gpu).detach().cpu()]
        total[nam] += time.time() - start
        # if nam=="R_over_S":
        #     print(est[nam])
        #     input()

for nam, runtime in total.items():
    print(f"{nam}:", runtime)

x_range, deviations_lst, biases_lst, deviation_aucs, bias_aucs = calc_dev(
    [orig]*len(models),
    list(est.values()),
    0, 1, 1000, 0.05
)

for i, nam in enumerate(models.keys()):
    if nam=="R_over_S":
        biases_lst[i]=[val if (val<0.045 and val>-0.045) else None for val in biases_lst[0]]

general_plot({
    "Ys": biases_lst,
    "Xs": x_range,
    "xlabel": "Hurst",
    "ylabel": "bias",
    "title": "",
    "fname": "fBm_Hurst_biases_3200",
    "dirname": "./plots",
    "markers": None,
    "legend": {
        "location": "top_left",
        "labels": list(models.keys())
    },
    "matplotlib": {
        "calc_xtics": False,
        "width": 6, #6.7,
        "height": 3.63, #3.6,
        "style": "default"
    },
    # "colors":  ["red","red","red","red"]+[Category10[10][1],Category10[10][0]],
    # "dashes": ["dotted","dashdot","dashed","solid","solid","solid"],
    "color_settings": {
        "bg_transparent": False
    }
})

general_plot({
    "Ys": deviations_lst,
    "Xs": x_range,
    "xlabel": "Hurst",
    "ylabel": "deviation",
    "title": "",
    "fname": "fBm_Hurst_deviations_3200",
    "dirname": "./plots",
    "markers": None,
    "legend": {
        "location": "top_left",
        "labels": list(models.keys())
    },
    "matplotlib": {
        "calc_xtics": False,
        "width": 6,#6.7,
        "height": 3.63,#3.6,
        "style": "default"
    },
    # "colors":  ["red","red","red","red"]+[Category10[10][1],Category10[10][0]],
    # "dashes": ["dotted","dashdot","dashed","solid","solid","solid"],
    "color_settings": {
        "bg_transparent": False
    }
})

scatter_grid = [{
    "Xs": orig,
    "Ys": Ys,
    "xlabel": "real H",
    "ylabel": "inferred H",
    #"title": title,
    "fname": f"fBm_Hurst_3200_scatter_grid",
    "dirname": "./plots",
    "circle_size": 10,
    "opacity": 0.3,
    "colors": [Category10[10][i]],
    "line45_color": "black",
    "legend": {
        "location": "bottom_right",
        "labels": [nam],
        "markerscale": 2.0
    },
    "matplotlib": {
        "width": 6,
        "height": 6,
        "style": "default"
    }
} for i, (nam, Ys) in enumerate(est.items())]
scatter_grid_plot(
    params_list=scatter_grid,
    width=2,
    export_types=["png", "pdf"],
    make_subfolder=True
)

scatter_grid = [{
    "Xs": orig,
    "Ys": [y-x for x, y in zip(orig,Ys)],
    "xlabel": "H",
    "ylabel": "error",
    #"title": title,
    "fname": f"fBm_Hurst_3200_error_scatter_grid",
    "dirname": "./plots",
    "circle_size": 10,
    "opacity": 0.3,
    "colors": [Category10[10][i]],
    "line45_color": None,
    "legend": {
        "location": "bottom_right",
        "labels": [nam],
        "markerscale": 2.0
    },
    "matplotlib": {
        "width": 6,
        "height": 6,
        "style": "default"
    }
} for i, (nam, Ys) in enumerate(est.items())]
scatter_grid_plot(
    params_list=scatter_grid,
    width=2,
    export_types=["png", "pdf"],
    make_subfolder=True
)
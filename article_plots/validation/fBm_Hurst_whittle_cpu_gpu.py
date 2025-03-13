import numpy as  np
import os
import logging
import random
import math
import torch
from bokeh.palettes import Category10
from tqdm import trange

import sys
sys.path.append('..')
sys.path.append('../..')
from metrics.plotters import general_plot, scatter_plot
from metrics.deviation import calc_dev
from process_generators.fbm_gen import gen as fbm_gen
from models.baselines.whittle import Model as Whittle
from models.baselines.whittle_gpu import Model as Whittle_gpu
import time

def to_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var

diff = True
num_cores = 32
whittle = Whittle({'diff': diff, "num_cores": num_cores})
#whittle_gpu = Whittle_gpu({'diff': diff, "num_cores": num_cores})
whittle_gpu = torch.nn.DataParallel(to_cuda(Whittle_gpu({'diff': diff, "num_cores": num_cores})))

orig = []
est = []
est_whittle = []
est_whittle_gpu = []

total_cpu = 0
total_gpu = 0
pbar=trange(1)
for _ in pbar:
    pbar.set_description("Generating.")
    inputs = []
    for _ in range(1000):
        H = random.uniform(0, 1)
        orig.append(H)        
        process = fbm_gen(hurst = H, n = 3200)
        inputs.append(np.asarray(process))

    start = time.time()
    pbar.set_description("whittle GPU")
    est_whittle_gpu += [float(val) for val in whittle_gpu(to_cuda(torch.FloatTensor(np.asarray(inputs)))).detach().cpu()]
    total_gpu += time.time() - start

    start = time.time()
    pbar.set_description("whittle CPU")
    est_whittle += [float(val) for val in whittle(inputs)]
    total_cpu += time.time() - start

print("whittle CPU:", total_cpu)
print("whittle GPU:", total_gpu)

x_range, deviations_lst, biases_lst, deviation_aucs, bias_aucs = calc_dev(
    [orig]*2,
    [est_whittle, est_whittle_gpu],
    0, 1, 1000, 0.1
)

general_plot({
    "Ys": biases_lst,
    "Xs": x_range,
    "xlabel": "Hurst",
    "ylabel": "bias",
    "title": "",
    "fname": "fBm_Hurst_Whittle_biases",
    "dirname": "./plots",
    "markers": None,
    "legend": {
        "location": "top_left",
        "labels": ["Whittle CPU","Whittle GPU"]
    },
    "matplotlib": {
        "calc_xtics": False,
        "width": 6, #6.7,
        "height": 3.63, #3.6,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})

general_plot({
    "Ys": deviations_lst,
    "Xs": x_range,
    "xlabel": "Hurst",
    "ylabel": "deviation",
    "title": "",
    "fname": "fBm_Hurst_Whittle_deviations",
    "dirname": "./plots",
    "markers": None,
    "legend": {
        "location": "top_left",
        "labels": ["Whittle CPU","Whittle GPU"]
    },
    "matplotlib": {
        "calc_xtics": False,
        "width": 6,#6.7,
        "height": 3.63,#3.6,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})

scatter_plot({
    "Xs": [est_whittle],
    "Ys": [est_whittle_gpu],
    "xlabel": "Whittle CPU",
    "ylabel": "Whittle GPU",
    "title": "",
    "fname": "fBm_Hurst_Whittle_scatter",
    "dirname": "./plots",
    "circle_size": 5,
    "x_jitter": 0,
    "opacity": 0,
    "heatmap": False,
    "line45_color": None,
    "legend": {
        "labels": ["Whittle"],
        "location": "top_left",
        "markerscale": 2.
    },
    "matplotlib": {
        "width": 6,
        "height": 6,
        # "width": 9,
        # "height": 4.75,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})

scatter_plot({
    "Xs": [orig],
    "Ys": [list(np.asarray(est_whittle) - np.asarray(est_whittle_gpu))],
    "xlabel": "H",
    "ylabel": "H_CPU - H_GPU",
    "ylim": [min(list(np.asarray(est_whittle) - np.asarray(est_whittle_gpu)))*1.05, max(list(np.asarray(est_whittle) - np.asarray(est_whittle_gpu)))*1.05],
    "title": "",
    "fname": "fBm_Hurst_Whittle_scatter_diff",
    "dirname": "./plots",
    "circle_size": 5,
    "x_jitter": 0,
    "opacity": 0,
    "heatmap": False,
    "line45_color": None,
    "legend": {
        "labels": ["Whittle"],
        "location": "bottom_right",
        "markerscale": 2.
    },
    "matplotlib": {
        "width": 6,
        "height": 6,
        # "width": 9,
        # "height": 4.75,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})

scatter_plot({
    "Xs": [orig]*2,
    "Ys": [est_whittle, est_whittle_gpu],
    "xlabel": "H",
    "ylabel": "diff",
    "title": "",
    "fname": "fBm_Hurst_Whittle_scatter_true_vs_ests",
    "dirname": "./plots",
    "circle_size": 5,
    "x_jitter": 0,
    "opacity": 0,
    "heatmap": False,
    "line45_color": "black",
    "legend": {
        "labels": ["Whittle CPU", "Whittle GPU"],
        "location": "top_left",
        "markerscale": 2.
    },
    "matplotlib": {
        "width": 6,
        "height": 6,
        # "width": 9,
        # "height": 4.75,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})
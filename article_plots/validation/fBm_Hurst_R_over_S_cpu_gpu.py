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
from models.baselines.R_over_S import Model as R_over_S
from models.baselines.R_over_S_gpu import Model as R_over_S_gpu
import time

def to_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var

diff = True
num_cores = 32
R_over_S = R_over_S({'diff': diff, "num_cores": num_cores})
R_over_S_gpu = torch.nn.DataParallel(to_cuda(R_over_S_gpu({'diff': diff, "num_cores": num_cores})))

orig = []
est = []
est_R_over_S = []
est_R_over_S_gpu = []

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
    pbar.set_description("R_over_S GPU")
    est_R_over_S_gpu += [float(val) for val in R_over_S_gpu(to_cuda(torch.FloatTensor(np.asarray(inputs)))).detach().cpu()]
    total_gpu += time.time() - start

    start = time.time()
    pbar.set_description("R_over_S CPU")
    est_R_over_S += [float(val) for val in R_over_S(inputs)]
    total_cpu += time.time() - start

print("R_over_S CPU:", total_cpu)
print("R_over_S GPU:", total_gpu)

x_range, deviations_lst, biases_lst, deviation_aucs, bias_aucs = calc_dev(
    [orig]*2,
    [est_R_over_S, est_R_over_S_gpu],
    0, 1, 1000, 0.1
)

general_plot({
    "Ys": biases_lst,
    "Xs": x_range,
    "xlabel": "Hurst",
    "ylabel": "bias",
    "title": "",
    "fname": "fBm_Hurst_R_over_S_biases",
    "dirname": "./plots",
    "markers": None,
    "legend": {
        "location": "top_left",
        "labels": ["R_over_S CPU","R_over_S GPU"]
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
}, export_types=["png", "pdf"])

general_plot({
    "Ys": deviations_lst,
    "Xs": x_range,
    "xlabel": "Hurst",
    "ylabel": "deviation",
    "title": "",
    "fname": "fBm_Hurst_R_over_S_deviations",
    "dirname": "./plots",
    "markers": None,
    "legend": {
        "location": "top_left",
        "labels": ["R_over_S CPU","R_over_S GPU"]
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
}, export_types=["png", "pdf"])

scatter_plot({
    "Xs": [est_R_over_S],
    "Ys": [est_R_over_S_gpu],
    "xlabel": "R_over_S CPU",
    "ylabel": "R_over_S GPU",
    "title": "",
    "fname": "fBm_Hurst_R_over_S_scatter",
    "dirname": "./plots",
    "circle_size": 5,
    "x_jitter": 0,
    "opacity": 0,
    "heatmap": False,
    "line45_color": None,
    "legend": {
        "labels": ["R_over_S"],
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
}, export_types=["png", "pdf"])

scatter_plot({
    "Xs": [orig],
    "Ys": [list(np.asarray(est_R_over_S) - np.asarray(est_R_over_S_gpu))],
    "xlabel": "H",
    "ylabel": "H_CPU - H_GPU",
    "ylim": [min(list(np.asarray(est_R_over_S) - np.asarray(est_R_over_S_gpu)))*1.05, max(list(np.asarray(est_R_over_S) - np.asarray(est_R_over_S_gpu)))*1.05],
    "title": "",
    "fname": "fBm_Hurst_R_over_S_scatter_diff",
    "dirname": "./plots",
    "circle_size": 5,
    "x_jitter": 0,
    "opacity": 0,
    "heatmap": False,
    "line45_color": None,
    "legend": {
        "labels": ["R_over_S"],
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
}, export_types=["png", "pdf"])

scatter_plot({
    "Xs": [orig]*2,
    "Ys": [est_R_over_S, est_R_over_S_gpu],
    "xlabel": "H",
    "ylabel": "estimated",
    "title": "",
    "fname": "fBm_Hurst_R_over_S_scatter_true_vs_ests",
    "dirname": "./plots",
    "circle_size": 5,
    "x_jitter": 0,
    "opacity": 0,
    "heatmap": False,
    "line45_color": "black",
    "permute": False,
    "legend": {
        "labels": ["R_over_S CPU", "R_over_S GPU"],
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
}, export_types=["png", "pdf"])
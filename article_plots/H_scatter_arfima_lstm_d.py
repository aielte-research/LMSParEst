import numpy as np
import torch
from bokeh.palettes import Category10
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm

from load_models import load_models, to_cuda
import sys
sys.path.append('../')
from process_generators.arfima_gen import gen as arfima_gen
from metrics.plotters import scatter_plot, general_plot
from metrics.deviation import calc_dev

models = {est:model for est,model in zip(["r_over_s", "variogram", "higuchi", "whittle", "lstm", "conv1d"],load_models(cuda=True))}

hursts = list(np.random.uniform(0, 1, 10000))

processes = [np.cumsum(arfima_gen(d = H-0.5, n = 12800)) for H in hursts]

Ys = {est: [] for est in ["lstm", "conv1d"]}

for process in tqdm(processes):
    input=to_cuda(torch.FloatTensor(np.asarray([process])))
    for est in ["lstm", "conv1d"]:
        estimate = [float(val[0]) for val in models[est](input).detach().cpu()][0]
        Ys[est].append(estimate)

def estimate_step(process):
    input=torch.FloatTensor(np.asarray([process]))
    ret={}
    for est in ["r_over_s", "variogram", "higuchi", "whittle"]:
        estimate = [float(val[0]) for val in models[est](input).detach().cpu()][0]
        ret[est] = estimate
    return ret

with ProcessPool(nodes=25) as pool:
    output_dicts = list(tqdm(pool.imap(estimate_step, processes), total=len(processes)))

# Initialize the final result dictionary
for est in ["r_over_s", "variogram", "higuchi", "whittle"]:
    Ys[est] = []

# Aggregate the results
for output in output_dicts:
    for key in ["r_over_s", "variogram", "higuchi", "whittle"]:
        Ys[key].append(output[key])

Ys = [Ys[key] for key in [ "r_over_s", "variogram", "higuchi", "whittle", "conv1d", "lstm"]]
methods = ["R/S","variogram","Higuchi","Whittle","1D CNN","LSTM"]

scatter_plot({
    "Xs": [hursts]*6,
    "Ys": Ys,
    "xlabel": "d + 1/2",
    "ylabel": "Inferred Value",
    "title": "",
    "fname": "H_scatter_arfima_lstm_d",
    "dirname": "./plots",
    "circle_size": 5,
    "x_jitter": 0,
    "opacity": 0.5,
    "heatmap": False,
    "line45_color": "black",
    "legend": {
        "labels": methods,
        "location": "top_left",
        "markerscale": 2.
    },
    "matplotlib": {
        "width": 6,
        "height": 3.5,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})

steps = 250
measure_interval = 0.02

x_range, deviations_lst, biases_lst, deviation_aucs, bias_aucs = calc_dev([hursts]*6, Ys, 0, 1, steps, measure_interval)

legend_labels = [f'{nam} (AUC:{auc:.3g})' for nam,auc in zip(methods,deviation_aucs)]
general_plot({
    "Ys": deviations_lst,
    "Xs": x_range,
    "xlabel": "d + 1/2",
    "ylabel": "deviation",
    "title": "Deviations (steps: {}, interval:+-{})".format(steps,measure_interval),
    "fname": "H_scatter_arfima_lstm_d_deviations",
    "dirname": "./plots",
    "markers": None,
    "colors":  ["red","red","red","red"]+[Category10[10][1],Category10[10][0]],
    "dashes": ["dotted","dashdot","dashed","solid","solid","solid"],
    "legend":{
        "location": "top_left", # "top_left", "top_right", "bottom_left", "bottom_right", None
        "labels": legend_labels
    },
    "matplotlib": {
        "width": 6,
        "height": 3.5,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})

legend_labels = [f'{nam} (AUC:{auc:.3g})' for nam,auc in zip(methods,bias_aucs)]
general_plot({
    "Ys": biases_lst,
    "Xs": x_range,
    "xlabel": "d + 1/2",
    "ylabel": "bias",
    "title":  "Biases (steps: {}, interval:+-{})".format(steps,measure_interval),
    "fname": "H_scatter_arfima_lstm_d_biases",
    "dirname": "./plots",
    "markers": None,
    "colors":  ["red","red","red","red"]+[Category10[10][1],Category10[10][0]],
    "dashes": ["dotted","dashdot","dashed","solid","solid","solid"],
    "legend":{
        "location": "top_right", # "top_left", "top_right", "bottom_left", "bottom_right", None
        "labels": legend_labels
    },
    "matplotlib": {
        "width": 6,
        "height": 3.5,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})
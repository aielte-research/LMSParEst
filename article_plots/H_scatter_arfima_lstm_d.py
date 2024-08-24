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
from models.baselines.whittle import Model as Whittle

models = {est:model for est,model in zip([
    "r_over_s",
    "variogram",
    "higuchi",
    "whittle",
    "lstm",
    "conv1d"
],load_models(cuda=True))}
models["whittle_arfima"] = Whittle({'num_cores':1, 'spec_name': "arfima"},None)

ds = np.random.uniform(-0.5, 0.5, 5000)

print("Generating process realizations...")
processes = [np.cumsum(arfima_gen(d = d, n = 12800)) for d in tqdm(ds)]

Ys = {est: [] for est in ["lstm", "conv1d"]}

print("Estimating parameters with neural models...")
for process in tqdm(processes):
    input=to_cuda(torch.FloatTensor(np.asarray([process])))
    for est in ["lstm", "conv1d"]:
        estimate = [float(val[0]) for val in models[est](input).detach().cpu()][0]
        Ys[est].append(estimate)

baselines = ["r_over_s", "variogram", "higuchi", "whittle", "whittle_arfima"]
def estimate_step(process):
    input=torch.FloatTensor(np.asarray([process]))
    ret={}
    for est in baselines:
        estimate = [float(val[0]) for val in models[est](input).detach().cpu()][0]
        ret[est] = estimate
    return ret

print("Estimating parameters with baseline models...")
with ProcessPool(nodes=25) as pool:
    output_dicts = list(tqdm(pool.imap(estimate_step, processes), total=len(processes)))

# Initialize the final result dictionary
for est in baselines:
    Ys[est] = []

# Aggregate the results
for output in output_dicts:
    for key in baselines:
        Ys[key].append(output[key])

Ys = [Ys[key] for key in baselines + ["conv1d", "lstm"]]
labels = ["R/S","variogram","Higuchi","Whittle fBm","Whittle ARFIMA","1D CNN","LSTM"]

Xs = list(ds+0.5)

scatter_plot({
    "Xs": [Xs],#*2,
    "Ys": Ys[-1:],
    "xlabel": "d + 1/2",
    "ylabel": "Inferred Value",
    "title": "",
    "fname": "H_scatter_arfima_lstm_d",
    "dirname": "./plots",
    "circle_size": 5,
    "x_jitter": 0,
    "opacity": 0,
    "heatmap": False,
    "line45_color": "black",
    # "colors": [Category10[10][1],Category10[10][0]],
    # "legend": {
    #     "labels": labels[-2:],
    #     "location": "top_left",
    #     "markerscale": 2.
    # },
    "matplotlib": {
        "width": 6.7,
        "height": 3.6,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})

steps = 250
measure_interval = 0.02

x_range, deviations_lst, biases_lst, deviation_aucs, bias_aucs = calc_dev([Xs]*7, Ys, 0, 1, steps, measure_interval)

legend_labels = [f'{nam} (AUC:{auc:.3g})' for nam,auc in zip(labels,deviation_aucs)]
general_plot({
    "Ys": deviations_lst,
    "Xs": x_range,
    "xlabel": "d + 1/2",
    "ylabel": "deviation",
    "title": "Deviations (steps: {}, interval:+-{})".format(steps,measure_interval),
    "fname": "H_scatter_arfima_lstm_d_deviations",
    "dirname": "./plots",
    "markers": None,
    "colors":  ["red","red","red","red","grey"]+[Category10[10][1],Category10[10][0]],
    "dashes": ["dotted","dashdot","dashed","solid","solid","solid","solid"],
    "legend":{
        "location": "top_left", # "top_left", "top_right", "bottom_left", "bottom_right", None
        "labels": legend_labels
    },
    "matplotlib": {
        "width": 6.7,
        "height": 3.6,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})

legend_labels = [f'{nam} (AUC:{auc:.3g})' for nam,auc in zip(labels,bias_aucs)]
general_plot({
    "Ys": biases_lst,
    "Xs": x_range,
    "xlabel": "d + 1/2",
    "ylabel": "bias",
    "title":  "Biases (steps: {}, interval:+-{})".format(steps,measure_interval),
    "fname": "H_scatter_arfima_lstm_d_biases",
    "dirname": "./plots",
    "markers": None,
    "colors":  ["red","red","red","red","grey"]+[Category10[10][1],Category10[10][0]],
    "dashes": ["dotted","dashdot","dashed","solid","solid","solid","solid"],
    "legend":{
        "location": "top_right", # "top_left", "top_right", "bottom_left", "bottom_right", None
        "labels": legend_labels
    },
    "matplotlib": {
        "width": 6.7,
        "height": 3.6,
        "style": "default"
    },
    "color_settings":{
        "bg_transparent": False
    }
})
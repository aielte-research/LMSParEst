import numpy as np
import torch
from bokeh.palettes import Category10
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm

from load_models import load_models, to_cuda
import sys
sys.path.append('../')
from process_generators.fbm_gen import gen as fbm_gen
from metrics.plotters import scatter_plot, general_plot
from metrics.deviation import calc_dev

from models.baselines.R_over_S import Model as R_over_S
from models.baselines.variogram import Model as Variogram
from models.baselines.higuchi import Model as Higuchi
from models.baselines.whittle import Model as Whittle
from models.LSTM import Model as LSTM
def to_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var

print(torch.cuda.is_available())

model_params={
    "diff": False,
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

state_dict_path="../model_checkpoints/ARFIMA/ARFIMA_d_LSTM_finetune_until_n-12800.pt"
lstm = LSTM(model_params, state_dict_path)
lstm = to_cuda(lstm)
lstm.eval()
models = {"lstm": lstm}

hursts = np.random.uniform(0, 1, 5000)

print("Generating process realizations...")
processes = [np.diff(fbm_gen(hurst = H, n = 12800)) for H in tqdm(hursts)]

Ys = {est: [] for est in ["lstm"]}

print("Estimating parameters with neural models...")
for process in tqdm(processes):
    input=to_cuda(torch.FloatTensor(np.asarray([process])))
    for est in ["lstm"]:
        estimate = [float(val[0]) for val in models[est](input).detach().cpu()][0]
        Ys[est].append(estimate)

diff = False
models["r_over_s"] = R_over_S({'diff':diff},None)
models["variogram"] = Variogram({'diff':diff,'num_cores':1},None)
models["higuchi"] = Higuchi({'diff':diff},None)
models["whittle_fGn"] = Whittle({'diff':diff,'num_cores':1, 'spec_name': "fgn"},None)
models["whittle"] = Whittle({'diff':diff,'num_cores':1, 'spec_name': "arfima"},None)

baselines = ["r_over_s", "variogram", "higuchi", "whittle_fGn", "whittle"]
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
        Ys[key].append(output[key] - 0.5)

Ys = [Ys[key] for key in baselines + ["lstm"]]
labels = ["R/S", "variogram", "Higuchi", "Whittle fBm", "Whittle ARFIMA", "LSTM"]

Xs = list(hursts - 0.5)
scatter_plot({
    "Xs": [Xs],#*2,
    "Ys": Ys[-1],
    "xlabel": "H - 1/2",
    "ylabel": "Inferred Value",
    "title": "",
    "fname": "d_scatter_lstm_arfima_H",
    "dirname": "./plots",
    "circle_size": 7,
    "x_jitter": 0,
    "opacity": 0,
    "heatmap": False,
    # "colors": [Category10[10][3],Category10[10][0]],
    "line45_color": "black",
    # "legend": {
    #     "labels": ["Whittle", "LSTM"], #labels[-2:],
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

x_range, deviations_lst, biases_lst, deviation_aucs, bias_aucs = calc_dev([Xs]*6, Ys, -0.5, 0.5, steps, measure_interval)

legend_labels = [f'{nam} (AUC:{auc:.3g})' for nam,auc in zip(labels,deviation_aucs)]
general_plot({
    "Ys": deviations_lst,
    "Xs": x_range,
    "xlabel": "H - 1/2",
    "ylabel": "deviation",
    "title": "Deviations (steps: {}, interval:+-{})".format(steps,measure_interval),
    "fname": "d_scatter_lstm_arfima_H_deviations",
    "dirname": "./plots",
    "markers": None,
    "colors": ["grey"]*4+["red"]+[Category10[10][0]],
    "dashes": ["dotted","dashdot","dashed","solid","solid","solid"],
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
    "xlabel": "H - 1/2",
    "ylabel": "bias",
    "title":  "Biases (steps: {}, interval:+-{})".format(steps,measure_interval),
    "fname": "d_scatter_lstm_arfima_H_biases",
    "dirname": "./plots",
    "markers": None,
    "colors": ["grey"]*4+["red"]+[Category10[10][0]],
    "dashes": ["dotted","dashdot","dashed","solid","solid","solid"],
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
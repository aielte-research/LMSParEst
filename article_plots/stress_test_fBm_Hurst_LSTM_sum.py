import random 
import torch
import numpy as np
from bokeh.palettes import Category10
from tqdm import tqdm

import sys
sys.path.append('..')
from metrics.plotters import general_plot, scatter_plot, scatter_grid_plot
from process_generators.fbm_gen import gen as fbm_gen
from models.LSTM import Model as LSTM
from models.baselines.R_over_S import Model as R_over_S
from models.baselines.variogram import Model as Variogram
from models.baselines.higuchi import Model as Higuchi
from models.baselines.whittle import Model as Whittle

# Setup estimators and parameters.
diff = True
num_cores = 32
r_over_s = R_over_S({'diff': diff, "num_cores": num_cores}, None)
variogram = Variogram({'diff': diff, "num_cores": num_cores}, None)
higuchi = Higuchi({'diff': diff, "num_cores": num_cores}, None)
whittle = Whittle({'diff': diff, "num_cores": num_cores}, None)

def to_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var

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

state_dict_path = "../model_checkpoints/fBm/fBm_Hurst_LSTM_finetune_until_n-12800.pt"
lstm = to_cuda(LSTM(model_params, state_dict_path))
lstm.eval()

# ----------------------------
# Experimental settings:
# Fixed H1 values.
H1_values = [0.1, 0.3, 0.5, 0.7, 0.9]

# Define the 3 combination variants.
def variant_sum(p1, p2):
    # Elementwise sum of the two processes.
    return p1 + p2

def variant_concat(p1, p2):
    # Concatenate the two processes (resulting length 3200).
    return np.concatenate((p1, p2))

def variant_fade(p1, p2):
    # Fade the two processes together via a convex combination.
    # For a process of length L, at index i the weight for p1 is:
    #     weight_p1 = i/(L-1)
    # and for p2 is:
    #     weight_p2 = 1 - i/(L-1)
    # This yields: combined_signal = weight_p1 * p1 + weight_p2 * p2.
    L = len(p1)
    weights = np.linspace(0, 1, L)      # weight for p1: 0 at index 0, 1 at index L-1
    weights_inv = 1 - weights           # weight for p2: 1 at index 0, 0 at index L-1
    return weights * p1 + weights_inv * p2

variants = {
    "sum": variant_sum,
    "concat": variant_concat,
    "fade": variant_fade
}

# ----------------------------
# Define batched estimator functions that accept a numpy array
# (with shape (batch_size, sequence_length)) of signals.
def estimate_lstm_batch(signals):
    tensor = torch.FloatTensor(signals)
    tensor = to_cuda(tensor)
    outputs = lstm(tensor).detach().cpu().numpy().flatten()
    return outputs

def estimate_r_over_s_batch(signals):
    tensor = torch.FloatTensor(signals)
    return np.array([float(val) for val in r_over_s(tensor.cpu())])

def estimate_variogram_batch(signals):
    tensor = torch.FloatTensor(signals)
    return np.array([float(val) for val in variogram(tensor.cpu())])

def estimate_higuchi_batch(signals):
    tensor = torch.FloatTensor(signals)
    return np.array([float(val) for val in higuchi(tensor.cpu())])

def estimate_whittle_batch(signals):
    tensor = torch.FloatTensor(signals)
    return np.array([float(val) for val in whittle(tensor.cpu())])

batch_estimators = {
    "R/S": estimate_r_over_s_batch,
    "variogram": estimate_variogram_batch,
    "Higuchi": estimate_higuchi_batch,
    "Whittle": estimate_whittle_batch,
    "LSTM": estimate_lstm_batch
}

# ----------------------------
# Prepare a data structure for scatter plots.
# For each variant and estimator, for each fixed H1 we store lists of H2 and estimated H.
scatter_data = {
    variant: {
        est: {h: {"H2": [], "est": []} for h in H1_values}
        for est in batch_estimators.keys()
    }
    for variant in variants.keys()
}

# Batch size (number of experiments per fixed H1 per batch iteration)
batch_size = 1000
# Number of batch iterations per H1 value.
num_batch_iterations = 1

# Loop over each fixed H1 value.
for H1 in tqdm(H1_values, desc="H1 values"):
    for batch_iter in range(num_batch_iterations):
        H2_batch = []
        process1_batch = []
        process2_batch = []
        # Generate a batch of fBm processes: fixed H1 and a random H2.
        for _ in range(batch_size):
            H2 = random.uniform(0, 1)
            H2_batch.append(H2)
            p1 = fbm_gen(hurst=H1, n=1600)
            p2 = fbm_gen(hurst=H2, n=1600)
            process1_batch.append(p1)
            process2_batch.append(p2)
            
        # For each combination variant, combine the batch of processes.
        for variant_name, combine_func in variants.items():
            combined_batch = []
            for p1, p2 in zip(process1_batch, process2_batch):
                combined_signal = combine_func(p1, p2)
                combined_batch.append(combined_signal)
            # Stack to form a 2D array: shape (batch_size, sequence_length)
            combined_batch = np.stack(combined_batch)
            
            # Run each estimator on the entire batch.
            for est_name, est_func in batch_estimators.items():
                est_values = est_func(combined_batch)
                # Store the H2 values and corresponding estimated H values.
                scatter_data[variant_name][est_name][H1]["H2"].extend(H2_batch)
                scatter_data[variant_name][est_name][H1]["est"].extend(est_values.tolist())

# ----------------------------
# Build scatter plot parameters for each (variant, estimator) pair.
# There will be 15 scatter plots (5 estimators x 3 variants).
custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#000000"]

scatter_params_list = []
for variant_name in variants.keys():
    for est_name in batch_estimators.keys():
        Xs_groups = []
        Ys_groups = []
        legend_labels = []
        for H1 in H1_values:
            Xs_groups.append(scatter_data[variant_name][est_name][H1]["H2"])
            Ys_groups.append(scatter_data[variant_name][est_name][H1]["est"])
            legend_labels.append(f"H₁ = {H1}")
        params = {
            "Xs": Xs_groups,
            "Ys": Ys_groups,
            "xlabel": "H₂",
            "ylabel": "Estimated H",
            "title": f"{est_name} estimator - {variant_name} variant",
            "fname": f"stress_test_scatter_fBm_Hurst_LSTM_sum",
            "dirname": "./plots",
            "circle_size": 4,
            "x_jitter": 0,
            "opacity": 0,
            "heatmap": False,
            "colors": custom_palette,
            "line45_color": "black",
            "legend": {
                "location": "bottom_right",
                "labels": legend_labels,
                "markerscale": 2.0
            },
            "matplotlib": {
                "width": 6,
                "height": 6,
                "style": "default"
            },
            "color_settings": {
                "bg_transparent": False
            }
        }
        scatter_params_list.append(params)

# ----------------------------
# Produce the grid of scatter plots.
scatter_grid_plot(scatter_params_list, width=5)

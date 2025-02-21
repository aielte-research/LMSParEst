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

# Define the experiments: original 1-10 chunks plus extra experiments for 20 and 50 chunks.
n_chunks_list = [1, 2, 4, 8, 16, 32, 64, 100, 200, 400, 800, 1600]

# For the line plot (MSE vs. number of chunks)
Xs = n_chunks_list[:]  # x-axis for the general plot
Ys = []           # MSE for LSTM
Ys_r_over_s = []  # MSE for R/S
Ys_variogram = [] # MSE for variogram
Ys_higuchi = []   # MSE for Higuchi
Ys_whittle = []   # MSE for Whittle

# Build a list of parameter dictionaries for the scatter grid plot.
scatter_params_list = []

# Loop over each experiment (different number of chunks)
for n_chunks in tqdm(n_chunks_list):
    orig = []
    est = []
    est_r_over_s = []
    est_variogram = []
    est_higuchi = []
    est_whittle = []
    
    # We'll also store scatter data from each experiment
    scatter_Xs = []  # true Hurst values

    # Run a few repetitions per experiment.
    for _ in range(100):
        inputs = []
        # Use a smaller sample for each repetition.
        for __ in range(1000):
            H = random.uniform(0, 1)
            orig.append(H)
            scatter_Xs.append(H)
            
            # Generate the fBm realization.
            process = fbm_gen(hurst=H, n=1600)
            if n_chunks > 1:
                # Split the process into nearly equal chunks and randomly permute the chunks.
                chunks = np.array_split(process, n_chunks)
                random.shuffle(chunks)
                permuted_process = np.concatenate(chunks)
            else:
                permuted_process = process

            inputs.append(np.asarray(permuted_process))
        
        input_tensor = to_cuda(torch.FloatTensor(inputs))
        # Compute estimations from different estimators.
        est += [float(val[0]) for val in lstm(input_tensor).detach().cpu()]
        est_r_over_s += [float(val) for val in r_over_s(input_tensor.cpu())]
        est_variogram += [float(val) for val in variogram(input_tensor.cpu())]
        est_higuchi += [float(val) for val in higuchi(input_tensor.cpu())]
        est_whittle += [float(val) for val in whittle(input_tensor.cpu())]
    
    # Compute MSE for each estimator.
    mse_lstm = np.square(np.asarray(orig) - np.asarray(est)).mean()
    mse_r_over_s = np.square(np.asarray(orig) - np.asarray(est_r_over_s)).mean()
    mse_variogram = np.square(np.asarray(orig) - np.asarray(est_variogram)).mean()
    mse_higuchi = np.square(np.asarray(orig) - np.asarray(est_higuchi)).mean()
    mse_whittle = np.square(np.asarray(orig) - np.asarray(est_whittle)).mean()
    
    Ys.append(mse_lstm)
    Ys_r_over_s.append(mse_r_over_s)
    Ys_variogram.append(mse_variogram)
    Ys_higuchi.append(mse_higuchi)
    Ys_whittle.append(mse_whittle)
    
    # Build the scatter data arrays.
    scatter_Ys_lstm = list(est)
    scatter_Ys_r_over_s = list(est_r_over_s)
    scatter_Ys_variogram = list(est_variogram)
    scatter_Ys_higuchi = list(est_higuchi)
    scatter_Ys_whittle = list(est_whittle)
    
    # Build the parameter dictionary for this experiment.
    params = {
        "Xs": [scatter_Xs[:1000]] * 5,  # repeat the true Hurst values for each estimator (limit to first 2000 points)
        "Ys": [
            scatter_Ys_r_over_s[:1000],
            scatter_Ys_variogram[:1000],
            scatter_Ys_higuchi[:1000],
            scatter_Ys_whittle[:1000],
            scatter_Ys_lstm[:1000]
        ],
        "xlabel": "Hurst",
        "ylabel": "Inferred value",
        "title": f"n_chunks = {n_chunks}",
        "fname": f"stress_test_scatter_fBm_Hurst_LSTM_permutation",
        "dirname": "./plots",
        "circle_size": 4,
        "x_jitter": 0,
        "opacity": 0,
        "heatmap": False,
        "colors": [
            Category10[10][1],
            Category10[10][2],
            Category10[10][3],
            Category10[10][4],
            Category10[10][0]
        ],
        "line45_color": "black",
        "legend": {
            "location": "bottom_right",
            "labels": ["R/S", "variogram", "Higuchi", "Whittle", "LSTM"],
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

# Plot the overall MSE loss vs. number of chunks.
general_plot({
    "Ys": [Ys_r_over_s, Ys_variogram, Ys_higuchi, Ys_whittle, Ys],
    "Xs": Xs,
    "xlabel": "Number of Chunks",
    "ylabel": "MSE loss",
    "xscale": "log",
    "yscale": "log",
    "title": "",
    "fname": "stress_test_fBm_Hurst_LSTM_permutation",
    "dirname": "./plots",
    "legend": {
        "location": "bottom_right",
        "labels": ["R/S", "variogram", "Higuchi", "Whittle", "LSTM"]
    },
    "markers": None,
    "colors": ["red", "red", "red", "red"] + [Category10[10][0]],
    "dashes": ["dotted", "dashdot", "dashed", "solid", "solid"],
    "line45_color": None,
    "matplotlib": {
        "width": 6,
        "height": 4.5,
        "style": "default",
        "xtics": Xs
    },
    "color_settings": {
        "bg_transparent": False
    }
})

# Supply the built parameter list to the scatter grid plotter.
scatter_grid_plot(scatter_params_list, width=4)

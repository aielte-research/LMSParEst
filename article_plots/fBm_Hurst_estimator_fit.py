import os
import logging
import random
import math
import torch
import numpy as np
import pandas as pd
from bokeh.palettes import Category10
from tqdm import trange

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from matplotlib.colors import to_rgb, to_hex
import cvxpy as cp
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.feature_selection import mutual_info_regression
import dcor
from minepy import MINE
from scipy.stats import kendalltau

import sys

sys.path.append('..')
from process_generators.fbm_gen import gen as fbm_gen
from models.LSTM import Model as LSTM
from models.baselines.R_over_S import Model as R_over_S
from models.baselines.variogram import Model as Variogram
from models.baselines.higuchi import Model as Higuchi
from models.baselines.whittle import Model as Whittle
from models.baselines.autocovariance import Model as Autocov
from metrics.plotters import scatter_plot, scatter_grid_plot, area_plot, area_grid_plot

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d (%(funcName)s) - %(message)s"
)
logger = logging.getLogger(__name__)

ROOT_DIR = "estimator_fit"
DIFF = True
NUM_CORES = 38
BIN_SIZE = 0.01
NUM_PROCESSES_PER_BIN = 50
N = 400
STATE_DICT_PATH = "../model_checkpoints/fBm/fBm_Hurst_LSTM_finetune_until_n-400.pt"
MODEL_PARAMS = {
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
        "activation": {
            "name": "PReLU"
        }
    }
}

def get_color_mapping(model_names):
    """Map model names to colors (using Category10)."""
    return {name: Category10[10][i % len(Category10[10])] for i, name in enumerate(model_names)}

def to_cuda(var):
    """Move variable to CUDA if available."""
    return var.cuda() if torch.cuda.is_available() else var

def get_custom_bin_defs(bin_size):
    """
    Returns a list of tuples (bin_label, condition_fn) for splitting the [0,1] range.
    An extra bin for H==0 is included.
    """
    edges = np.arange(0, 1 + bin_size, bin_size)
    bin_defs = [(0.0, lambda x: (x == 0))]
    for i in range(1, len(edges)):
        low, high = edges[i - 1], edges[i]
        if i == 1:
            cond = lambda x, low=low, high=high: (x > low) & (x < high)
        elif i == len(edges) - 1:
            cond = lambda x, low=low, high=high: (x >= low) & (x <= high)
        else:
            cond = lambda x, low=low, high=high: (x >= low) & (x < high)
        bin_defs.append((high, cond))
    return bin_defs

def generate_fbm_processes(num_processes_per_bin, n):
    """
    Generate fBm processes with H uniformly drawn in each bin.
    (The extra H==0 part is commented out.)
    """
    logger.info(f"Generating fBm processes: {num_processes_per_bin} per bin, BIN_SIZE={BIN_SIZE}, n={n}")
    orig, processes = [], []
    bins = np.arange(0, 1 + BIN_SIZE, BIN_SIZE)
    logger.debug(f"Interior bins: {bins}")
    for i in trange(len(bins) - 1, desc="Generating interior processes"):
        lower, upper = bins[i], bins[i + 1]
        for _ in range(num_processes_per_bin):
            H = random.uniform(lower, upper)
            orig.append(H)
            processes.append(np.asarray(fbm_gen(hurst=H, n=n)))
    logger.info(f"Generated a total of {len(processes)} processes.")
    return orig, processes

def estimate_hurst_exponents_by_bin(orig, processes, models):
    """
    For each process, estimate the Hurst exponent using all models.
    """
    logger.info(f"Estimating Hurst exponents (BIN_SIZE={BIN_SIZE}).")
    batch_size = 200
    predictions = {name: [] for name in models}
    total = len(processes)
    for start in trange(0, total, batch_size, desc="Estimating processes"):
        batch = np.array(processes[start:start + batch_size])
        input_tensor = to_cuda(torch.FloatTensor(batch))
        for name, model in models.items():
            if name == "LSTM":
                pred = model(input_tensor).detach().cpu().numpy()
            else:
                pred = model(input_tensor.cpu())
                if torch.is_tensor(pred):
                    pred = pred.numpy()
            if pred.ndim == 2:
                predictions[name].extend(float(val[0]) for val in pred)
            else:
                predictions[name].extend(float(val) for val in pred)
    logger.info("Hurst exponent estimation completed.")
    return predictions

def plot_scatter(predictions, orig, color_mapping, title, xlabel="Hurst", ylabels=None, ylabel="Inferred value"):
    if ylabels is None:
        ylabels = {model: "Inferred value" for model in predictions.keys()}

    scatter_grid = [
        {
            "Xs": orig,
            "Ys": Ys,
            "xlabel": xlabel,
            "ylabel": ylabels[model],
            #"title": title,
            "fname": f"{title.replace(' ','_')}_scatter_grid",
            "dirname": "./estimator_fit/scatter",
            "circle_size": 10,
            "opacity": 0.3,
            "colors": [color_mapping[model]],
            "line45_color": "black",
            "legend": {
                "location": "bottom_right",
                "labels": [model],
                "markerscale": 2.0
            },
            "matplotlib": {
                "width": 6,
                "height": 6,
                "style": "default"
            }
        } for model, Ys in predictions.items()
    ]
    scatter_grid_plot(
        params_list=scatter_grid,
        width=math.ceil(len(predictions) / 2),
        export_types=["png", "pdf"],
        make_subfolder=True
    )

    scatter_options = scatter_grid[0].copy()
    scatter_options["Ys"] = list(predictions.values())
    scatter_options["ylabel"] = ylabel
    scatter_options["fname"] = f"{title.replace(' ','_')}_scatter"
    scatter_options["colors"] = [color_mapping[model] for model in predictions.keys()]
    scatter_options["legend"]["labels"] = list(predictions.keys())
    scatter_plot(scatter_options, export_types=["png", "pdf"])

    for scatter in scatter_grid:
        scatter["heatmap"] = True
        scatter["fname"] = f"{title.replace(' ','_')}_heatmap_grid"

    scatter_grid_plot(
        params_list=scatter_grid,
        width=math.ceil(len(predictions) / 2),
        export_types=["png", "pdf"],
        make_subfolder=True
    )

def linreg_callback(X_feat, y_resp):
    model = LinearRegression(fit_intercept=False).fit(X_feat, y_resp)
    coef = model.coef_
    # Remove normalization from here; raw coefficients are passed
    rmse = np.sqrt(np.mean((y_resp - model.predict(X_feat))**2))
    return (coef.tolist(), rmse)

def slsqp_callback(X_feat, y_resp):
    w = cp.Variable(X_feat.shape[1], nonneg=True)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(X_feat@w - y_resp)), [cp.sum(w) == 1])
    problem.solve()
    w_val = w.value
    rmse = np.sqrt(np.mean((y_resp - X_feat@w_val)**2))
    return (w_val.tolist(), rmse)

def _iterate_bins(orig, predictions, bin_defs, baseline, per_bin_callback):
    """
    Iterates over custom bins (each defined by a condition function) and applies a callback.
    """
    feature_names = [name for name in predictions if name != baseline]
    x = np.array(orig)
    y_features = np.column_stack([predictions[name] for name in feature_names])
    y_target = np.array(predictions[baseline])
    results = []
    bin_labels = []
    for (label, cond_fn) in bin_defs:
        mask = cond_fn(x)
        if not np.any(mask):
            continue
        X_feat = y_features[mask, :]
        y_resp = y_target[mask]
        results.append(per_bin_callback(X_feat, y_resp))
        bin_labels.append(label)
    return feature_names, bin_labels, results

def plot_regression(
    orig,
    predictions,
    color_mapping,
    baseline="LSTM",
    regression_callback=linreg_callback,
    regression_name="Linear Regression",
    normalized=True
):
    logger.info(f"Plotting Linear Regression stacked area plot (BIN_SIZE={BIN_SIZE}).")

    bin_defs = get_custom_bin_defs(BIN_SIZE)
    feature_names, bin_labels, results = _iterate_bins(orig, predictions, bin_defs, baseline, regression_callback)
    weights = [res[0] for res in results]
    rmse_vals = [res[1] for res in results]
    Ys = [[row[j] for row in weights] for j in range(len(feature_names))]

    fname = f"{baseline}_Hurst_{regression_name.replace(' ','_').lower()}"
    title = f'{regression_name} weights'
    ylabel = "Weights"
    if normalized:
        fname += "_normalized"
        title += " (normalized)"
        ylabel += " (normalized)"

    area_plot(dict(
        X=bin_labels,
        Ys=Ys,
        fname=fname,
        dirname=os.path.join(ROOT_DIR, "regression"),
        title=title,
        xlabel="Hurst exponent",
        ylabel=ylabel,
        normalized=normalized,
        colors=[color_mapping[name] for name in feature_names],
        legend_labels=feature_names,
        measure={
            "Y": rmse_vals,
            "label": 'RMSE',
            "color": "black",
            "dash": "dashed"
        }
    ), export_types=["pdf"], make_subfolder=False)

def plot_pca(
    data,
    true_values,
    color_mapping,
    n_components=4,
    title_prefix="PCA",
    fname="PCA"
):
    models_list = list(data.keys())
    pc_loadings = [[] for _ in range(n_components)]
    pc_explained = [[] for _ in range(n_components)]
    bin_labels = []
    bin_defs = get_custom_bin_defs(BIN_SIZE)

    for (label, cond_fn) in bin_defs:
        indices = np.where(cond_fn(np.array(true_values)))[0]
        if len(indices) == 0:
            continue
        data_list = []
        for model in models_list:
            transformed = np.array(data[model])[indices]
            data_list.append(transformed)
        X_bin = np.stack(data_list, axis=1)
        # Zscore normalization: subtract mean and divide by std for each column (i.e. each model)
        X_bin = (X_bin - np.mean(X_bin, axis=0)) / np.std(X_bin, axis=0)
        pca = PCA(n_components=n_components)
        pca.fit(X_bin)
        for comp_index in range(n_components):
            component = pca.components_[comp_index]
            pc_loadings[comp_index].append(component)
            pc_explained[comp_index].append(pca.explained_variance_ratio_[comp_index] * 100)
        bin_labels.append(label)

    plot_param_list = []
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        weights_matrix = pc_loadings[comp_index]  # shape: (num_bins, num_models)
        Ys = [[row[j] for row in weights_matrix] for j in range(len(models_list))]
        plot_param_list.append(dict(
            X=bin_labels,
            Ys=Ys,
            fname=fname,
            dirname=os.path.join(ROOT_DIR, "PCA"),
            title=f'{title_prefix} - Principal Component {comp_number}',
            xlabel="Hurst exponent",
            ylabel=f'PC{comp_number} Loading',
            normalized=True,
            colors=[color_mapping[name] for name in models_list],
            legend_labels=models_list,
            measure={
                "Y": pc_explained[comp_index],
                "label": 'Explained Variance (%)',
                "color": "black",
                "dash": "dashed"
            }
        ))
    area_grid_plot(plot_param_list, export_types=["pdf"], make_subfolder=False)
    logger.info(f"{title_prefix} PCA plots saved.")

def calc_cov(x, y):
    """Return the covariance between x and y."""
    if len(x) < 2:
        return 0
    return np.cov(x, y)[0, 1]

def calc_distance_corr(x, y):
    """Return the distance correlation between x and y."""
    if len(x) < 2:
        return 0
    return dcor.distance_correlation(np.array(x), np.array(y))

def calc_mic(x, y):
    """Return the maximal information coefficient (MIC) between x and y."""
    if len(x) < 2:
        return 0
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(np.array(x), np.array(y))
    return mine.mic()

def calc_slope(x, y):
    """Return the slope of y on x computed as cov(x, y) / var(y)."""
    if len(x) < 2:
        return 0
    var_y = np.var(y, ddof=1)
    if var_y == 0:
        return 0
    covariance = np.cov(x, y, ddof=1)[0, 1]
    return covariance / var_y

def calc_kendall_tau(x, y):
    """Return the Kendall's tau correlation between x and y."""
    if len(x) < 2:
        return 0
    tau, _ = kendalltau(x, y)
    return tau if not np.isnan(tau) else 0

def calc_blomqvist_beta(x, y):
    """Return Blomqvist's beta (quadrant correlation) between x and y."""
    if len(x) < 2:
        return 0
    median_x = np.median(x)
    median_y = np.median(y)
    same_quadrant = ((x-median_x) * (y-median_y)) > 0
    beta = 2 * np.mean(same_quadrant) - 1
    return beta

def calc_pearson(x, y):
    """Return the Pearson correlation between x and y."""
    if len(x) < 2:
        return 0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0
    r, _ = pearsonr(x, y)
    return r

def calc_spearman(x, y):
    """Return the Spearman correlation between x and y."""
    if len(x) < 2:
        return 0
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x == 0 or std_y == 0:
        return 0
    result = spearmanr(x, y).correlation
    if np.isnan(result):
        return 0
    return result

def calc_mutual_info(x, y):
    """Return the average mutual information between x and y."""
    if len(x) < 2:
        return 0
    mi1 = mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0]
    mi2 = mutual_info_regression(y.reshape(-1, 1), x, random_state=0)[0]
    return (mi1+mi2) / 2

def plot_measures_grid(target, predictions, orig, measures, color_mapping, suffix="", normalize=True):
    orig_arr = np.array(orig)
    bin_defs = get_custom_bin_defs(BIN_SIZE)

    plot_param_list = []

    for measure_name, measure_func in measures:
        measure_data = {other: [] for other in predictions if other != target}
        bin_labels = []
        for (label, cond_fn) in bin_defs:
            indices = np.where(cond_fn(orig_arr))[0]
            if len(indices) == 0:
                continue
            bin_labels.append(label)
            target_vals = np.array(predictions[target])[indices]
            for other in predictions:
                if other == target:
                    continue
                other_vals = np.array(predictions[other])[indices]
                val = measure_func(target_vals, other_vals)
                measure_data[other].append(val)
        plot_params = dict(
            X=bin_labels,
            Ys=list(measure_data.values()),
            fname=f"{target}_vs_others_{measure_name}{suffix}",
            dirname=os.path.join(ROOT_DIR, "CorrMtx", target,"normalized" if normalize else "original"),
            title=f'{target} vs others{suffix.replace("_"," ")}',
            xlabel="Hurst exponent",
            ylabel=f"{measure_name}{' (normalized)' if normalize else ''}",
            normalized=normalize,
            colors=[color_mapping[name] for name in measure_data.keys()],
            legend_labels=list(measure_data.keys())
        )

        area_plot(plot_params, export_types=["pdf"], make_subfolder=False)

        plot_params["fname"]=f"_{target}_vs_others{suffix}_grid"

        plot_param_list.append(plot_params)

    area_grid_plot(plot_param_list, width=3, export_types=["pdf"], make_subfolder=False)
    
# =============================================================================
# Model setup and run estimation
# =============================================================================

def setup_models():
    logger.info("Initializing models for Hurst exponent estimation.")
    lstm = to_cuda(LSTM(MODEL_PARAMS, STATE_DICT_PATH))
    lstm.eval()
    model_settings = {'diff': True, "num_cores": NUM_CORES}
    return {
        "LSTM": lstm,
        "R_over_S": R_over_S(model_settings, None),
        "variogram": Variogram(model_settings, None),
        "Higuchi": Higuchi(model_settings, None),
        "Whittle": Whittle(model_settings, None),
        # "autocov": Autocov({'diff': True, "num_cores": NUM_CORES}, None)
    }

def run_estimation(models):
    logger.info("Starting Hurst exponent estimation process.")
    orig, processes = generate_fbm_processes(NUM_PROCESSES_PER_BIN, n=N)
    predictions = estimate_hurst_exponents_by_bin(orig, processes, models)
    errors_dict = {model: np.array(predictions[model]) - np.array(orig) for model in predictions}
    mapping = get_color_mapping(list(predictions.keys()))

    logger.info("########################")
    logger.info("# Plotting scatters... #")
    logger.info("########################")
    for baseline in models.keys():
        logger.info(f"Scatter: {baseline} vs others")
        pred = {name: pred for name, pred in predictions.items() if name != baseline}
        plot_scatter(
            pred,
            predictions[baseline],
            mapping,
            f"{baseline} vs others",
            xlabel=f"Hurst by {baseline}",
            ylabels={model: f"Hurst by {model}" for model in pred.keys()}
        )

        logger.info(f"Scatter: {baseline} errors vs others")
        pred_err = {name: err for name, err in errors_dict.items() if name != baseline}
        plot_scatter(
            pred_err,
            errors_dict[baseline],
            mapping,
            f"{baseline} errors vs other errors",
            xlabel=f"{baseline} Error",
            ylabels={model: f"{model} Error" for model in pred_err.keys()},
            ylabel="Error"
        )

    logger.info(f"Scatter: true vs models")
    plot_scatter(predictions, orig, mapping, "true vs predicted Hurst")

    logger.info("##########################")
    logger.info("# Plotting Regression... #")
    logger.info("##########################")

    plot_regression(orig, predictions, color_mapping=mapping, regression_callback=linreg_callback, regression_name="Linear Regression")
    plot_regression(orig, predictions, color_mapping=mapping, regression_callback=linreg_callback, regression_name="Linear Regression", normalized=False)
    plot_regression(orig, predictions, color_mapping=mapping, regression_callback=slsqp_callback, regression_name="SLSQP")

    logger.info("###################")
    logger.info("# Plotting PCA... #")
    logger.info("###################")
    # PCA on raw predictions
    plot_pca(
        data=predictions,
        true_values=orig,
        color_mapping=mapping,
        n_components=4,
        title_prefix="PCA",
        fname="PCA"
    )
    # PCA on differences: (model - LSTM) for models other than LSTM
    plot_pca(
        data={m: np.array(vals) - np.array(predictions["LSTM"]) for m, vals in predictions.items() if m != "LSTM"},
        true_values=orig,
        color_mapping=mapping,
        n_components=4,
        title_prefix="PCA Diff (Model - LSTM)",
        fname="PCA_diff"
    )
    # PCA on errors: (predicted - true)
    plot_pca(
        data={m: np.array(vals) - np.array(orig) for m, vals in predictions.items()},
        true_values=orig,
        color_mapping=mapping,
        n_components=4,
        title_prefix="PCA Error",
        fname="PCA_error"
    )

    logger.info("############################")
    logger.info("# Plotting Correlations... #")
    logger.info("############################")
    # Measure plots (both non-normalized and normalized)
    measures = [
        ("Covariance", calc_cov),
        ("Pearson Correlation", calc_pearson),
        ("Spearman Correlation", calc_spearman),
        ("Mutual Information", calc_mutual_info),
        ("Slope", calc_slope),
        #("Distance_Correlation", calc_distance_corr),
        ("Maximal Information Coefficient", calc_mic),
        #("Kendalls_tau", calc_kendall_tau),
        #("Blomqvists_beta", calc_blomqvist_beta),
    ]

    # Grid plots for each target (raw and normalized)
    for target in predictions.keys():
        logger.info(f"Plotting correlations with {target}")
        for normalize in [False, True]:
            plot_measures_grid(target, predictions, orig, measures, color_mapping=mapping, normalize=normalize)
            plot_measures_grid(target, {m: np.array(vals) - np.array(orig) for m, vals in predictions.items()}, orig, measures, color_mapping=mapping, suffix="_errors", normalize=normalize)

if __name__ == "__main__":
    run_estimation(setup_models())
    logger.info("Process finished successfully.")
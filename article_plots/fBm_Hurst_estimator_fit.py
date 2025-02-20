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
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import cvxpy as cp
from matplotlib.patches import Patch
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression

import sys
sys.path.append('..')
from process_generators.fbm_gen import gen as fbm_gen
from models.LSTM import Model as LSTM
from models.baselines.R_over_S import Model as R_over_S
from models.baselines.variogram import Model as Variogram
from models.baselines.higuchi import Model as Higuchi
from models.baselines.whittle import Model as Whittle
from models.baselines.autocovariance import Model as Autocov

# =============================================================================

# Configure logging with the given format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d (%(funcName)s) - %(message)s"
)
logger = logging.getLogger(__name__)

# Global parameters
ROOT_DIR = "estimator_fit"
DIFF = True
NUM_CORES = 38
BIN_SIZE = 0.01
NUM_PROCESSES_PER_BIN = 10000
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
        "activation": {"name": "PReLU"}
    }
}

def get_color_mapping(model_names):
    """
    Automatically generate a mapping from model names to colors based on the order of the names.
    Colors are taken from Category10[10], with modulo applied if more than 10 models.
    """
    return {name: Category10[10][i % len(Category10[10])] for i, name in enumerate(model_names)}

def to_cuda(var):
    """Move variable to CUDA if available."""
    return var.cuda() if torch.cuda.is_available() else var

def darken_color(hex_color, factor=0.7):
    """Generate a darker tone for the given hex color."""
    logger.debug(f"Darkening color {hex_color} with factor {factor}.")
    rgb = np.array(to_rgb(hex_color))
    dark_hex = to_hex(rgb * factor)
    logger.debug(f"Resulting darkened color: {dark_hex}")
    return dark_hex

def get_custom_bin_defs(bin_size):
    """
    Returns a list of tuples (bin_label, condition_fn) that split the [0,1] range into:
      - an extra bin for H==0,
      - interior bins for values in (0,1] (with width = bin_size).

    For interior bins, the label is set to the right boundary (i.e. a full bin shift).
    """
    edges = np.arange(0, 1 + bin_size, bin_size)
    bin_defs = []
    bin_defs.append((0.0, lambda x: (x == 0)))
    for i in range(1, len(edges)):
        low = edges[i - 1]
        high = edges[i]
        if i == 1:
            cond = lambda x, low=low, high=high: (x > low) & (x < high)
        elif i == len(edges) - 1:
            cond = lambda x, low=low, high=high: (x >= low) & (x <= high)
        else:
            cond = lambda x, low=low, high=high: (x >= low) & (x < high)
        label = high
        bin_defs.append((label, cond))
    return bin_defs

def _iterate_bins_custom(orig, predictions, bin_defs, baseline, per_bin_callback, desc):
    """
    Iterates over custom bins defined by bin_defs.
    bin_defs is a list of tuples: (bin_label, condition_fn) where condition_fn(x) returns a boolean mask.
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
        result = per_bin_callback(X_feat, y_resp)
        results.append(result)
        bin_labels.append(label)
    return feature_names, bin_labels, results

def generate_fbm_processes(num_processes_per_bin, n):
    """
    Generate fBm processes.
    For each interior bin in [0,1), generate processes with H uniformly drawn from that bin.
    Then, explicitly generate extra sets of processes with H==0.

    :return: Tuple (list of true Hurst parameters, list of generated processes)
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
    for _ in trange(num_processes_per_bin, desc="Generating processes for H=0"):
        orig.append(0.0)
        processes.append(np.asarray(fbm_gen(hurst=0.0, n=n)))
    logger.info(f"Generated a total of {len(processes)} processes.")
    return orig, processes

def estimate_hurst_exponents_by_bin(orig, processes, models):
    """
    Estimate Hurst exponents for all models by processing the fBm processes in custom bins.
    Uses the custom bin definitions so that there is one bin for H==0 and interior bins for H in (0,1].
    """
    logger.info(f"Estimating Hurst exponents by custom bins with BIN_SIZE={BIN_SIZE}")
    batch_size = 200
    predictions = {name: [] for name in models}
    total = len(processes)
    for start in trange(0, total, batch_size, desc="Estimating processes"):
        batch = np.array(processes[start:start + batch_size])
        input_tensor = to_cuda(torch.FloatTensor(batch))
        for name, model in models.items():
            if name == "lstm":
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

def compute_mse(orig, predictions):
    """Compute the Mean Squared Error (MSE) for each model."""
    logger.info("Computing MSE for all models.")
    mse = {}
    orig_arr = np.array(orig)
    for name, pred in predictions.items():
        mse[name] = np.mean((orig_arr - np.array(pred)) ** 2)
        logger.debug(f"MSE for {name}: {mse[name]}")
    return mse

def plot_hurst_vs_est(predictions, baseline="lstm", color_mapping=None):
    """Scatter plot comparing baseline predictions with others."""
    logger.info("Plotting scatter plots for model comparison.")
    baseline_pred = predictions[baseline]
    other_names = [name for name in predictions if name != baseline]
    n_plots = len(other_names)
    if n_plots == 0:
        logger.warning("No other models to compare with.")
        return
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for ax, name in zip(axes, other_names):
        ax.scatter(baseline_pred, predictions[name],
                   color=color_mapping[name], alpha=0.7, label=name)
        ax.plot([-0.4, 1.2], [-0.4, 1.2], "k--")
        ax.set_xlabel(f"Hurst by {baseline}")
        ax.set_ylabel(f"Hurst by {name}")
        ax.set_xlim([-0.4, 1.2])
        ax.set_ylim([-0.4, 1.2])
        ax.legend()
    for ax in axes[len(other_names):]:
        ax.set_visible(False)
    fig.suptitle("Comparison between Models", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_dir = os.path.join(ROOT_DIR, "regression")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "Models_vs_models_grid.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "Models_vs_models_grid.pdf"), dpi=300, bbox_inches='tight')
    logger.info("Scatter plots saved.")

def mse_loss(weights, X_features, y_response):
    """Compute RMSE using linear regression (no intercept)."""
    logger.info("Computing RMSE loss via linear regression.")
    model = LinearRegression(fit_intercept=False)
    model.fit(X_features, y_response)
    rmse = np.sqrt(np.mean((y_response - model.predict(X_features)) ** 2))
    logger.debug(f"Computed RMSE: {rmse}")
    return rmse

def constraint_sum_to_one(w):
    """Constraint: sum of weights equals 1."""
    return np.sum(w) - 1

def plot_stacked_area(
    weights_df,
    rmse_primary,
    title='',
    xlabel='Hurst exponent',
    ylabel='Weights',
    rmse_ylabel='RMSE',
    fpath='',
    color_mapping=None
):
    """
    Create and save a stacked area plot.
    The x–coordinates are taken as the bin labels produced by our custom bins.
    """
    primary_colors = [color_mapping[col] for col in weights_df.columns]
    logger.info(f"Creating plot '{title}'.")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.array(weights_df.index)
    abs_values = weights_df.abs().to_numpy()
    cumulative = np.cumsum(abs_values, axis=1)
    bottoms = np.hstack([np.zeros((abs_values.shape[0], 1)), cumulative[:, :-1]])
    signs = weights_df.apply(np.sign).to_numpy()
    for j, col in enumerate(weights_df.columns):
        pos_color = primary_colors[j]
        neg_color = darken_color(pos_color, 0.7)
        for i in range(len(x) - 1):
            seg_color = pos_color if signs[i, j] > 0 else neg_color
            x_seg = x[i:i + 2]
            bottom_seg = [bottoms[i, j], bottoms[i + 1, j]]
            top_seg = [cumulative[i, j], cumulative[i + 1, j]]
            ax1.fill_between(x_seg, bottom_seg, top_seg, color=seg_color, alpha=0.7)
    legend_handles = [Patch(facecolor=color_mapping[col], label=col) for col in weights_df.columns]
    final_handles = legend_handles[::-1]
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title, pad=42)
    if rmse_ylabel is not None:
        ax2 = ax1.twinx()
        ax2.plot(x, rmse_primary, "k--", linewidth=2, label=rmse_ylabel)
        ax2.set_ylabel(rmse_ylabel)
        dashed_handles, _ = ax2.get_legend_handles_labels()
        final_handles += dashed_handles
        if "explained variance" in rmse_ylabel.lower():
            ax2.set_ylim(0, 100)
    ax1.legend(final_handles, [h.get_label() for h in final_handles], ncol=len(final_handles), bbox_to_anchor=(0, 1), loc='lower left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    plt.savefig(fpath + ".png", bbox_inches='tight')
    plt.savefig(fpath + ".pdf", bbox_inches='tight')
    plt.close(fig)
    logger.info("Stacked area plot saved.")

def plot_stacked_area_ax(
    ax,
    weights_df,
    rmse_primary,
    title='',
    xlabel='Hurst exponent',
    ylabel='Weights',
    rmse_ylabel='RMSE',
    color_mapping=None
):
    """Plot a stacked area chart on a given axis (ax) using the same logic as plot_stacked_area."""
    primary_colors = [color_mapping[col] for col in weights_df.columns]
    secondary_colors = [darken_color(c, 0.7) for c in primary_colors]
    x = np.array(weights_df.index)
    abs_values = weights_df.abs().to_numpy()
    cumulative = np.cumsum(abs_values, axis=1)
    bottoms = np.hstack([np.zeros((abs_values.shape[0], 1)), cumulative[:, :-1]])
    signs = weights_df.apply(np.sign).to_numpy()
    for j, col in enumerate(weights_df.columns):
        pos_color = primary_colors[j]
        neg_color = secondary_colors[j]
        for i in range(len(x) - 1):
            seg_color = pos_color if signs[i, j] > 0 else neg_color
            x_seg = x[i:i + 2]
            bottom_seg = [bottoms[i, j], bottoms[i + 1, j]]
            top_seg = [cumulative[i, j], cumulative[i + 1, j]]
            ax.fill_between(x_seg, bottom_seg, top_seg, color=seg_color, alpha=0.7)
    if rmse_ylabel is not None:
        ax2 = ax.twinx()
        ax2.plot(x, rmse_primary, "k--", linewidth=2, label=rmse_ylabel)
        ax2.set_ylabel(rmse_ylabel)
        if "explained variance" in rmse_ylabel.lower():
            ax2.set_ylim(0, 100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=36)
    legend_handles = [Patch(facecolor=color_mapping[col], label=col) for col in weights_df.columns]
    if rmse_ylabel is not None:
        dashed_line = plt.Line2D([0], [0], color='k', linestyle='--', label=rmse_ylabel)
        legend_handles.append(dashed_line)
    final_handles = legend_handles[::-1]
    ax.legend(final_handles, [h.get_label() for h in final_handles], ncol=len(final_handles), bbox_to_anchor=(0, 1), loc='lower left')

def plot_stacked_area_SLSQP(orig, predictions, baseline="lstm", color_mapping=None):
    """Compute weights via SLSQP and plot using custom bins."""
    logger.info(f"Plotting SLSQP stacked area plot with custom bins (BIN_SIZE={BIN_SIZE})")
    def slsqp_callback(X_feat, y_resp):
        w = cp.Variable(X_feat.shape[1], nonneg=True)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(X_feat @ w - y_resp)), [cp.sum(w) == 1])
        problem.solve()
        w_val = w.value
        rmse = np.sqrt(np.mean((y_resp - X_feat @ w_val) ** 2))
        return (w_val.tolist(), rmse)
    bin_defs = get_custom_bin_defs(BIN_SIZE)
    feature_names, bin_labels, results = _iterate_bins_custom(orig, predictions, bin_defs, baseline, slsqp_callback, desc="SLSQP optimization")
    weights = [res[0] for res in results]
    rmse_vals = [res[1] for res in results]
    weights_df = pd.DataFrame(weights, columns=feature_names, index=bin_labels)
    plot_stacked_area(
        weights_df,
        rmse_vals,
        fpath=os.path.join(ROOT_DIR, "regression", "Hurst_stacked_plot_sum1"),
        color_mapping=color_mapping
    )

def plot_stacked_area_LinReg(orig, predictions, baseline="lstm", color_mapping=None):
    """Compute weights via LinReg and plot using custom bins."""
    logger.info(f"Plotting Linear Regression stacked area plot with custom bins (BIN_SIZE={BIN_SIZE})")
    def linreg_callback(X_feat, y_resp):
        model = LinearRegression(fit_intercept=False).fit(X_feat, y_resp)
        coef = model.coef_
        norm_coef = coef / np.abs(coef).sum()
        rmse = np.sqrt(np.mean((y_resp - model.predict(X_feat)) ** 2))
        return (norm_coef, rmse)
    bin_defs = get_custom_bin_defs(BIN_SIZE)
    feature_names, bin_labels, results = _iterate_bins_custom(orig, predictions, bin_defs, baseline, linreg_callback, desc="LinReg processing")
    weights = [res[0] for res in results]
    rmse_vals = [res[1] for res in results]
    weights_df = pd.DataFrame(weights, columns=feature_names, index=bin_labels)
    plot_stacked_area(
        weights_df,
        rmse_vals,
        title='LinReg weights normalized by abs sum',
        fpath=os.path.join(ROOT_DIR, "regression", "Hurst_stacked_plot_lin_reg"),
        color_mapping=color_mapping
    )

def plot_pca_components(orig, predictions, n_components=4, color_mapping=None):
    """
    Perform PCA on normalized predictions and plot the components using custom bins.
    Now uses n_components=4 and saves both individual plots and a 2x2 grid.
    """
    logger.info("Performing PCA on normalized predictions for each custom bin and plotting PCA components.")
    models_list = list(predictions.keys())
    pc_loadings = [[] for _ in range(n_components)]
    pc_explained = [[] for _ in range(n_components)]
    bin_labels = []
    orig_arr = np.array(orig)
    bin_defs = get_custom_bin_defs(BIN_SIZE)
    for (label, cond_fn) in bin_defs:
        indices = np.where(cond_fn(orig_arr))[0]
        if len(indices) == 0:
            continue
        data = []
        for model in models_list:
            model_preds = np.array(predictions[model])
            bin_preds = model_preds[indices]
            mean_val = np.mean(bin_preds)
            std_val = np.std(bin_preds)
            normalized = (bin_preds - mean_val) / std_val if std_val != 0 else bin_preds - mean_val
            data.append(normalized)
        X_bin = np.stack(data, axis=1)
        pca = PCA(n_components=n_components)
        pca.fit(X_bin)
        for comp_index in range(n_components):
            component = pca.components_[comp_index]
            norm_component = component / np.sum(np.abs(component))
            pc_loadings[comp_index].append(norm_component)
            pc_explained[comp_index].append(pca.explained_variance_ratio_[comp_index] * 100)
        bin_labels.append(label)
    # Save individual plots for each principal component
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_list, index=bin_labels)
        plot_stacked_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA - PC{comp_number} Eigenvector Components and Explained Variance',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Eigenvector Loading',
            rmse_ylabel='Explained Variance (%)',
            fpath=os.path.join(ROOT_DIR, "PCA", f"PCA_PC{comp_number}"),
            color_mapping=color_mapping
        )
    # Save grid plot (2x2) for all components with updated figure size
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_list, index=bin_labels)
        ax = axes[comp_index]
        plot_stacked_area_ax(
            ax,
            pc_df,
            pc_explained[comp_index],
            title=f'PCA - PC{comp_number} Components and Explained Variance',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            color_mapping=color_mapping
        )
    plt.tight_layout()
    save_dir = os.path.join(ROOT_DIR, "PCA")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "_PCA_components_grid.png"), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "_PCA_components_grid.pdf"), bbox_inches='tight')
    plt.close(fig)
    logger.info("PCA components plots saved (individual and grid).")

def plot_pca_diff_components(orig, predictions, n_components=4, color_mapping=None):
    """
    Perform PCA on differences (model - LSTM) and plot using custom bins.
    Saves both individual plots and a 2x2 grid.
    """
    logger.info("Performing PCA on normalized differences (model - LSTM) for each custom bin and plotting PCA diff components.")
    models_diff = [model for model in predictions.keys() if model != "lstm"]
    pc_loadings = [[] for _ in range(n_components)]
    pc_explained = [[] for _ in range(n_components)]
    bin_labels = []
    orig_arr = np.array(orig)
    lstm_preds_all = np.array(predictions["lstm"])
    bin_defs = get_custom_bin_defs(BIN_SIZE)
    for (label, cond_fn) in bin_defs:
        indices = np.where(cond_fn(orig_arr))[0]
        if len(indices) == 0:
            continue
        data = []
        lstm_bin = lstm_preds_all[indices]
        for model in models_diff:
            model_preds = np.array(predictions[model])
            bin_preds = model_preds[indices]
            diff = bin_preds - lstm_bin
            mean_val = np.mean(diff)
            std_val = np.std(diff)
            normalized = (diff - mean_val) / std_val if std_val != 0 else diff - mean_val
            data.append(normalized)
        X_bin = np.stack(data, axis=1)
        pca = PCA(n_components=n_components)
        pca.fit(X_bin)
        for comp_index in range(n_components):
            component = pca.components_[comp_index]
            norm_component = component / np.sum(np.abs(component))
            if norm_component[0] < 0:
                norm_component = -norm_component
            pc_loadings[comp_index].append(norm_component)
            pc_explained[comp_index].append(pca.explained_variance_ratio_[comp_index] * 100)
        bin_labels.append(label)
    # Save individual plots
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_diff, index=bin_labels)
        plot_stacked_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA Diff - PC{comp_number} (Model - LSTM) Components and Explained Variance',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            fpath=os.path.join(ROOT_DIR, "PCA", "diff", f"PCA_diff_PC{comp_number}"),
            color_mapping=color_mapping
        )
    # Save grid plot with updated figure size
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_diff, index=bin_labels)
        ax = axes[comp_index]
        plot_stacked_area_ax(
            ax,
            pc_df,
            pc_explained[comp_index],
            title=f'PCA Diff - PC{comp_number} Components and Explained Variance',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            color_mapping=color_mapping
        )
    plt.tight_layout()
    save_dir = os.path.join(ROOT_DIR, "PCA", "diff")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "_PCA_diff_components_grid.png"), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "_PCA_diff_components_grid.pdf"), bbox_inches='tight')
    plt.close(fig)
    logger.info("PCA difference components plots saved (individual and grid).")

def plot_pca_error_components(orig, predictions, n_components=4, color_mapping=None):
    """
    Perform PCA on errors (estimation - true) without z-scoring and plot using custom bins.
    Saves both individual plots and a 2x2 grid.
    """
    logger.info("Performing PCA on errors (estimated - true) for each custom bin and plotting PCA error components.")
    models_list = list(predictions.keys())
    pc_loadings = [[] for _ in range(n_components)]
    pc_explained = [[] for _ in range(n_components)]
    bin_labels = []
    orig_arr = np.array(orig)
    bin_defs = get_custom_bin_defs(BIN_SIZE)
    for (label, cond_fn) in bin_defs:
        indices = np.where(cond_fn(orig_arr))[0]
        if len(indices) == 0:
            continue
        data = []
        for model in models_list:
            model_preds = np.array(predictions[model])
            bin_preds = model_preds[indices]
            true_vals = orig_arr[indices]
            errors = bin_preds - true_vals
            data.append(errors)
        X_bin = np.stack(data, axis=1)
        pca = PCA(n_components=n_components)
        pca.fit(X_bin)
        for comp_index in range(n_components):
            component = pca.components_[comp_index]
            norm_component = component / np.sum(np.abs(component))
            if norm_component[0] < 0:
                norm_component = -norm_component
            pc_loadings[comp_index].append(norm_component)
            pc_explained[comp_index].append(pca.explained_variance_ratio_[comp_index] * 100)
        bin_labels.append(label)
    # Save individual plots
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_list, index=bin_labels)
        plot_stacked_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA Error - PC{comp_number} Components and Explained Variance',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            fpath=os.path.join(ROOT_DIR, "PCA", "error", f"PCA_error_PC{comp_number}"),
            color_mapping=color_mapping
        )
    # Save grid plot with updated figure size
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_list, index=bin_labels)
        ax = axes[comp_index]
        plot_stacked_area_ax(
            ax,
            pc_df,
            pc_explained[comp_index],
            title=f'PCA Error - PC{comp_number} Components and Explained Variance',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            color_mapping=color_mapping
        )
    plt.tight_layout()
    save_dir = os.path.join(ROOT_DIR, "PCA", "error")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "_PCA_error_components_grid.png"), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, "_PCA_error_components_grid.pdf"), bbox_inches='tight')
    plt.close(fig)
    logger.info("PCA error components plots saved (individual and grid).")
    
def calc_cov(x, y):
    """Return the covariance between x and y."""
    if len(x) < 2:
        return 0
    return np.cov(x, y)[0, 1]

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
    return (mi1 + mi2) / 2

def plot_measure_across_bins_single(predictions, orig, measure_func, measure_name, color_mapping=None):
    """
    For each target estimator, compute (for each custom bin) the measure between its predictions and those of each other estimator.
    The row is normalized to sum to 1, and then a stacked area plot is created.
    """
    orig_arr = np.array(orig)
    bin_defs = get_custom_bin_defs(BIN_SIZE)
    for target in predictions.keys():
        measure_data = {other: [] for other in predictions if other != target}
        bin_labels = []
        for (label, cond_fn) in bin_defs:
            indices = np.where(cond_fn(orig_arr))[0]
            if len(indices) == 0:
                continue
            bin_labels.append(label)
            target_vals = np.array(predictions[target])[indices]
            for other in predictions.keys():
                if other == target:
                    continue
                other_vals = np.array(predictions[other])[indices]
                val = measure_func(target_vals, other_vals)
                measure_data[other].append(val)
        if len(bin_labels) == 0:
            continue
        df = pd.DataFrame(measure_data, index=bin_labels)
        df_norm = df.div(df.sum(axis=1).replace(0, 1), axis=0)
        rmse_primary = [0] * len(df_norm.index)
        plot_stacked_area(
            df_norm,
            rmse_primary,
            title=f"{target} {measure_name} with Others",
            xlabel="Hurst Exponent Bin",
            ylabel=f"Normalized {measure_name}",
            rmse_ylabel=None,
            fpath=os.path.join(ROOT_DIR, "CorrMtx", target, f"{target}_{measure_name}_single"),
            color_mapping=color_mapping
        )

def plot_measures_grid(target, predictions, orig, diff=False, color_mapping=None):
    """
    For a given target model, compute for each custom bin the correlation/association measure 
    between its predictions (or errors if diff=True) and those of each other estimator.
    Four measures (Covariance, Pearson, Spearman, Mutual Information) are computed and arranged in a 2x2 grid.
    """
    measures = [
        ("Covariance", calc_cov),
        ("Pearson", calc_pearson),
        ("Spearman", calc_spearman),
        ("Mutual_Information", calc_mutual_info)
    ]
    orig_arr = np.array(orig)
    bin_defs = get_custom_bin_defs(BIN_SIZE)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    for idx, (measure_name, measure_func) in enumerate(measures):
        measure_data = {other: [] for other in predictions if other != target}
        bin_labels = []
        for (label, cond_fn) in bin_defs:
            indices = np.where(cond_fn(orig_arr))[0]
            if len(indices) == 0:
                continue
            bin_labels.append(label)
            target_vals = np.array(predictions[target])[indices]
            if diff:
                true_vals = np.array(orig)[indices]
                target_vals = target_vals - true_vals
            for other in predictions:
                if other == target:
                    continue
                other_vals = np.array(predictions[other])[indices]
                if diff:
                    true_vals = np.array(orig)[indices]
                    other_vals = other_vals - true_vals
                val = measure_func(target_vals, other_vals)
                measure_data[other].append(val)
        if len(bin_labels) == 0:
            continue
        df = pd.DataFrame(measure_data, index=bin_labels)
        df_norm = df.div(df.sum(axis=1).replace(0, 1), axis=0)
        rmse_primary = [0] * len(df_norm.index)
        ax = axes[idx]
        plot_stacked_area_ax(
            ax,
            df_norm,
            rmse_primary,
            title=f'{target} {measure_name} {"Differences" if diff else "Predictions"}',
            xlabel="Hurst Exponent Bin",
            ylabel=f"Normalized {measure_name}",
            rmse_ylabel=None,
            color_mapping=color_mapping
        )
    suptitle = f'{target} Measures {"Differences" if diff else "Predictions"}'
    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_dir = os.path.join(ROOT_DIR, "CorrMtx", target)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"_{target}_measures{'_diff' if diff else ''}_grid.png"), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f"_{target}_measures{'_diff' if diff else ''}_grid.pdf"), bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Plotted grid {suptitle}")

def setup_models():
    """Initialize and return models for Hurst exponent estimation."""
    logger.info("Initializing models for Hurst exponent estimation.")
    lstm = to_cuda(LSTM(MODEL_PARAMS, STATE_DICT_PATH))
    lstm.eval()
    return {
        "lstm": lstm,
        "r_over_s": R_over_S({'diff': True, "num_cores": NUM_CORES}, None),
        "variogram": Variogram({'diff': True, "num_cores": NUM_CORES}, None),
        "higuchi": Higuchi({'diff': True, "num_cores": NUM_CORES}, None),
        "whittle": Whittle({'diff': True, "num_cores": NUM_CORES}, None),
        # "autocov": Autocov({'diff': True, "num_cores": NUM_CORES}, None)
    }

def run_estimation(models):
    """Run the full estimation and plotting workflow."""
    logger.info("Starting Hurst exponent estimation process.")
    orig, processes = generate_fbm_processes(NUM_PROCESSES_PER_BIN, n=N)
    predictions = estimate_hurst_exponents_by_bin(orig, processes, models)
    mse_dict = compute_mse(orig, predictions)
    logger.info(f"MSE values: {mse_dict}")
    # Compute the color mapping once based on all prediction keys.
    mapping = get_color_mapping(list(predictions.keys()))
    # Scatter/regression plots
    plot_hurst_vs_est(predictions, baseline="lstm", color_mapping=mapping)
    plot_stacked_area_LinReg(orig, predictions, baseline="lstm", color_mapping=mapping)
    plot_stacked_area_SLSQP(orig, predictions, baseline="lstm", color_mapping=mapping)
    # PCA plots (individual and grid)
    plot_pca_components(orig, predictions, color_mapping=mapping)
    plot_pca_diff_components(orig, predictions, color_mapping=mapping)
    plot_pca_error_components(orig, predictions, color_mapping=mapping)
    # Measure plots: For each measure, save both single plots and grid plots.
    for measure_name, measure_func in [
        ("Covariance", calc_cov),
        ("Pearson", calc_pearson),
        ("Spearman", calc_spearman),
        ("Mutual_Information", calc_mutual_info)
    ]:
        plot_measure_across_bins_single(predictions, orig, measure_func, measure_name, color_mapping=mapping)
    # And also the grid versions for predictions and differences:
    for target in predictions.keys():
        plot_measures_grid(target, predictions, orig, diff=False, color_mapping=mapping)
        plot_measures_grid(target, predictions, orig, diff=True, color_mapping=mapping)
    logger.info("Hurst exponent estimation process completed.")

if __name__ == "__main__":
    run_estimation(setup_models())
    logger.info("Process finished successfully.")

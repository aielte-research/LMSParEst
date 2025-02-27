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
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.feature_selection import mutual_info_regression
import dcor
from minepy import MINE

import sys

sys.path.append('..')
from process_generators.fbm_gen import gen as fbm_gen
from models.LSTM import Model as LSTM
from models.baselines.R_over_S import Model as R_over_S
from models.baselines.variogram import Model as Variogram
from models.baselines.higuchi import Model as Higuchi
from models.baselines.whittle import Model as Whittle
from models.baselines.autocovariance import Model as Autocov
from metrics.plotters import scatter_plot, scatter_grid_plot

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d (%(funcName)s) - %(message)s"
)
logger = logging.getLogger(__name__)

ROOT_DIR = "estimator_fit"
DIFF = True
NUM_CORES = 38
BIN_SIZE = 0.01
NUM_PROCESSES_PER_BIN = 100
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

def darken_color(hex_color, factor=0.7):
    """Return a darker version of the given hex color."""
    logger.debug(f"Darkening color {hex_color} with factor {factor}.")
    rgb = np.array(to_rgb(hex_color))
    dark_hex = to_hex(rgb * factor)
    logger.debug(f"Resulting darkened color: {dark_hex}")
    return dark_hex

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

def _iterate_bins_custom(orig, predictions, bin_defs, baseline, per_bin_callback, desc):
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

def compute_mse(orig, predictions):
    """Compute the Mean Squared Error for each model."""
    logger.info("Computing MSE for all models.")
    mse = {}
    orig_arr = np.array(orig)
    for name, pred in predictions.items():
        mse[name] = np.mean((orig_arr - np.array(pred))**2)
        logger.debug(f"MSE for {name}: {mse[name]}")
    return mse

# =============================================================================
# Unified area plotting function
# =============================================================================

def plot_area(
    weights_df,
    rmse_primary,
    title='',
    xlabel='Hurst exponent',
    ylabel='Weights',
    rmse_ylabel='RMSE',
    fpath=None,
    color_mapping=None,
    normalized=False,
    stacked=True,
    ax=None
):
    """
    Unified area plotting function.
    
    Parameters:
      - weights_df: a DataFrame whose index will be used as x and whose columns are plotted.
      - rmse_primary: list/array of values to plot as a line (e.g. RMSE or explained variance).
      - title, xlabel, ylabel: text for plot.
      - rmse_ylabel: label for the secondary y-axis line; if None, no line is plotted.
      - fpath: if provided and a new figure is created, the plot is saved to fpath+".pdf".
      - color_mapping: dict mapping column names to colors.
      - normalized: if True, sets the y-axis limit to [0,1].
      - stacked: if True, plot a stacked area (using cumulative absolute values and sign-based colors);
                 if False, plot the raw (non-stacked) areas.
      - ax: if provided, plot on this axis; otherwise, create a new figure.
    """
    new_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        new_fig = True

    # Prepare x values from the DataFrame index.
    x = np.array(weights_df.index, dtype=float)
    x = np.where(x == 0, 0, x - (BIN_SIZE/2))

    cols = list(weights_df.columns)
    if color_mapping is None:
        color_mapping = {col: Category10[10][i % len(Category10[10])] for i, col in enumerate(cols)}

    if stacked:
        abs_values = weights_df.abs().to_numpy()
        cumulative = np.cumsum(abs_values, axis=1)
        bottoms = np.hstack([np.zeros((abs_values.shape[0], 1)), cumulative[:, :-1]])
        signs = weights_df.apply(np.sign).to_numpy()
        for j, col in enumerate(cols):
            pos_color = color_mapping[col]
            neg_color = darken_color(pos_color, 0.7)
            for i in range(len(x) - 1):
                seg_color = pos_color if signs[i, j] > 0 else neg_color
                ax.fill_between(
                    x[i:i + 2], [bottoms[i, j], bottoms[i + 1, j]], [cumulative[i, j], cumulative[i + 1, j]],
                    color=seg_color,
                    alpha=0.7
                )
    else:
        values = weights_df.to_numpy()
        for j, col in enumerate(cols):
            col_color = color_mapping[col]
            for i in range(len(x) - 1):
                seg_vals = [values[i, j], values[i + 1, j]]
                ax.fill_between(x[i:i + 2], [0, 0], seg_vals, color=col_color, alpha=0.33)
    # Plot the secondary line if required.
    if rmse_ylabel is not None:
        ax2 = ax.twinx()
        ax2.plot(x, rmse_primary, "k--", linewidth=2, label=rmse_ylabel)
        ax2.set_ylabel(rmse_ylabel)
        if "explained variance" in rmse_ylabel.lower():
            ax2.set_ylim(0, 100)
        rmse_handle = plt.Line2D([0], [0], color='k', linestyle='--', label=rmse_ylabel)
    # Set limits, labels, and title.
    ax.set_xlim(np.min(x), np.max(x))
    if normalized:
        ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    pad = 42 if new_fig else 36
    ax.set_title(title, pad=pad)
    # Build legend.
    legend_handles = [Patch(facecolor=color_mapping[col], label=col) for col in cols]
    if rmse_ylabel is not None:
        legend_handles.append(rmse_handle)
    ax.legend(
        legend_handles, [h.get_label() for h in legend_handles],
        ncol=len(legend_handles),
        bbox_to_anchor=(0, 1),
        loc='lower left'
    )

    if new_fig:
        plt.tight_layout()
        if fpath:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            plt.savefig(f"{fpath}.pdf", dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

# =============================================================================
# Scatter and regression plots (unchanged)
# =============================================================================

def plot_scatter(predictions, orig, color_mapping, title, xlabel="Hurst", ylabels=None, ylabel="Inferred value"):
    logger.info("Plotting scatter...")

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

    logger.info("Scatter plot saved.")

def mse_loss(weights, X_features, y_response):
    logger.info("Computing RMSE loss via linear regression.")
    model = LinearRegression(fit_intercept=False)
    model.fit(X_features, y_response)
    rmse = np.sqrt(np.mean((y_response - model.predict(X_features))**2))
    logger.debug(f"Computed RMSE: {rmse}")
    return rmse

# =============================================================================
# Unified area plot usage in regression and PCA plots
# =============================================================================

def plot_stacked_area_SLSQP(orig, predictions, baseline="LSTM", color_mapping=None):
    logger.info(f"Plotting SLSQP stacked area plot (BIN_SIZE={BIN_SIZE}).")

    def slsqp_callback(X_feat, y_resp):
        w = cp.Variable(X_feat.shape[1], nonneg=True)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(X_feat@w - y_resp)), [cp.sum(w) == 1])
        problem.solve()
        w_val = w.value
        rmse = np.sqrt(np.mean((y_resp - X_feat@w_val)**2))
        return (w_val.tolist(), rmse)

    bin_defs = get_custom_bin_defs(BIN_SIZE)
    feature_names, bin_labels, results = _iterate_bins_custom(
        orig, predictions, bin_defs, baseline, slsqp_callback, desc="SLSQP"
    )
    weights = [res[0] for res in results]
    rmse_vals = [res[1] for res in results]
    weights_df = pd.DataFrame(weights, columns=feature_names, index=bin_labels)
    fpath = os.path.join(ROOT_DIR, "regression", "Hurst_stacked_plot_sum1")
    plot_area(
        weights_df,
        rmse_vals,
        title='SLSQP',
        xlabel='Hurst exponent',
        ylabel='Weights',
        rmse_ylabel='RMSE',
        fpath=fpath,
        color_mapping=color_mapping,
        normalized=True,
        stacked=True
    )
    plot_area(
        weights_df,
        rmse_vals,
        title="SLSQP non-stacked",
        xlabel='Hurst exponent',
        ylabel='Weights',
        rmse_ylabel='RMSE',
        fpath=fpath + "_nonstacked",
        color_mapping=color_mapping,
        normalized=True,
        stacked=False
    )

def plot_stacked_area_LinReg(orig, predictions, baseline="LSTM", color_mapping=None):
    logger.info(f"Plotting Linear Regression stacked area plot (BIN_SIZE={BIN_SIZE}).")

    def linreg_callback(X_feat, y_resp):
        model = LinearRegression(fit_intercept=False).fit(X_feat, y_resp)
        coef = model.coef_
        norm_coef = coef / np.abs(coef).sum()
        rmse = np.sqrt(np.mean((y_resp - model.predict(X_feat))**2))
        return (norm_coef, rmse)

    bin_defs = get_custom_bin_defs(BIN_SIZE)
    feature_names, bin_labels, results = _iterate_bins_custom(
        orig, predictions, bin_defs, baseline, linreg_callback, desc="LinReg"
    )
    weights = [res[0] for res in results]
    rmse_vals = [res[1] for res in results]
    weights_df = pd.DataFrame(weights, columns=feature_names, index=bin_labels)
    fpath = os.path.join(ROOT_DIR, "regression", "Hurst_stacked_plot_lin_reg")
    # Stacked version
    plot_area(
        weights_df,
        rmse_vals,
        title='LinReg weights normalized by abs sum',
        xlabel='Hurst exponent',
        ylabel='Weights',
        rmse_ylabel='RMSE',
        fpath=fpath,
        color_mapping=color_mapping,
        normalized=True,
        stacked=True
    )
    # Non-stacked version
    plot_area(
        weights_df,
        rmse_vals,
        title='LinReg non-stacked weights normalized by abs sum',
        xlabel='Hurst exponent',
        ylabel='Weights',
        rmse_ylabel='RMSE',
        fpath=fpath + "_nonstacked",
        color_mapping=color_mapping,
        normalized=True,
        stacked=False
    )

def plot_pca_components(orig, predictions, n_components=4, color_mapping=None):
    logger.info("Performing PCA on normalized predictions and plotting components.")
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
            bin_preds = np.array(predictions[model])[indices]
            mean_val = np.mean(bin_preds)
            std_val = np.std(bin_preds)
            normalized_vals = (bin_preds-mean_val) / std_val if std_val != 0 else bin_preds - mean_val
            data.append(normalized_vals)
        X_bin = np.stack(data, axis=1)
        pca = PCA(n_components=n_components)
        pca.fit(X_bin)
        for comp_index in range(n_components):
            component = pca.components_[comp_index]
            norm_component = component / np.sum(np.abs(component))
            pc_loadings[comp_index].append(norm_component)
            pc_explained[comp_index].append(pca.explained_variance_ratio_[comp_index] * 100)
        bin_labels.append(label)
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_list, index=bin_labels)
        fpath = os.path.join(ROOT_DIR, "PCA", f"PCA_PC{comp_number}")
        # Stacked version
        plot_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA - PC{comp_number} Eigenvector Components and Explained Variance',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Eigenvector Loading',
            rmse_ylabel='Explained Variance (%)',
            fpath=fpath,
            color_mapping=color_mapping,
            normalized=True,
            stacked=True
        )
        # Non-stacked version
        plot_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA - PC{comp_number} Eigenvector Components and Explained Variance (non-stacked)',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Eigenvector Loading',
            rmse_ylabel='Explained Variance (%)',
            fpath=fpath + "_nonstacked",
            color_mapping=color_mapping,
            normalized=True,
            stacked=False
        )
    # Grid plots for PCA components (stacked)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_list, index=bin_labels)
        ax = axes[comp_index]
        plot_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA - PC{comp_number} Components and Explained Variance',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            color_mapping=color_mapping,
            normalized=True,
            stacked=True,
            ax=ax
        )
    plt.tight_layout()
    save_dir = os.path.join(ROOT_DIR, "PCA")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "_PCA_components_grid.pdf"), bbox_inches='tight')
    plt.close(fig)
    # Grid plots for PCA components (non-stacked)
    fig_ns, axes_ns = plt.subplots(2, 2, figsize=(20, 12))
    axes_ns = axes_ns.flatten()
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_list, index=bin_labels)
        ax = axes_ns[comp_index]
        plot_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA - PC{comp_number} Components and Explained Variance (non-stacked)',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            color_mapping=color_mapping,
            normalized=True,
            stacked=False,
            ax=ax
        )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "_PCA_components_grid_nonstacked.pdf"), bbox_inches='tight')
    plt.close(fig_ns)
    logger.info("PCA components plots saved.")

def plot_pca_diff_components(orig, predictions, n_components=4, color_mapping=None):
    logger.info("Performing PCA on normalized differences (model - LSTM) and plotting components.")
    models_diff = [m for m in predictions.keys() if m != "LSTM"]
    pc_loadings = [[] for _ in range(n_components)]
    pc_explained = [[] for _ in range(n_components)]
    bin_labels = []
    orig_arr = np.array(orig)
    lstm_preds_all = np.array(predictions["LSTM"])
    bin_defs = get_custom_bin_defs(BIN_SIZE)
    for (label, cond_fn) in bin_defs:
        indices = np.where(cond_fn(orig_arr))[0]
        if len(indices) == 0:
            continue
        data = []
        lstm_bin = lstm_preds_all[indices]
        for model in models_diff:
            bin_preds = np.array(predictions[model])[indices]
            diff = bin_preds - lstm_bin
            mean_val = np.mean(diff)
            std_val = np.std(diff)
            normalized_diff = (diff-mean_val) / std_val if std_val != 0 else diff - mean_val
            data.append(normalized_diff)
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
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_diff, index=bin_labels)
        # Stacked version
        plot_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA Diff - PC{comp_number} (Model - LSTM) Components and Explained Variance',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            fpath=os.path.join(ROOT_DIR, "PCA", "diff", "stacked", f"PCA_diff_PC{comp_number}"),
            color_mapping=color_mapping,
            normalized=True,
            stacked=True
        )
        # Non-stacked version
        plot_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA Diff - PC{comp_number} (Model - LSTM) Components and Explained Variance (non-stacked)',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            fpath=os.path.join(ROOT_DIR, "PCA", "diff", "nonstacked", f"PCA_diff_PC{comp_number}"),
            color_mapping=color_mapping,
            normalized=True,
            stacked=False
        )
    # Grid plots for PCA diff components (stacked)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_diff, index=bin_labels)
        ax = axes[comp_index]
        plot_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA Diff - PC{comp_number} Components and Explained Variance',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            color_mapping=color_mapping,
            normalized=True,
            stacked=True,
            ax=ax
        )
    plt.tight_layout()
    save_dir = os.path.join(ROOT_DIR, "PCA", "diff")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "_PCA_diff_components_grid.pdf"), bbox_inches='tight')
    plt.close(fig)
    # Grid plots for PCA diff components (non-stacked)
    fig_ns, axes_ns = plt.subplots(2, 2, figsize=(20, 12))
    axes_ns = axes_ns.flatten()
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_diff, index=bin_labels)
        ax = axes_ns[comp_index]
        plot_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA Diff - PC{comp_number} Components and Explained Variance (non-stacked)',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            color_mapping=color_mapping,
            normalized=True,
            stacked=False,
            ax=ax
        )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "_PCA_diff_components_grid_nonstacked.pdf"), bbox_inches='tight')
    plt.close(fig_ns)
    logger.info("PCA diff components plots saved.")

def plot_pca_error_components(orig, predictions, n_components=4, color_mapping=None):
    logger.info("Performing PCA on errors (estimated - true) and plotting components.")
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
            bin_preds = np.array(predictions[model])[indices]
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
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_list, index=bin_labels)
        fpath = os.path.join(ROOT_DIR, "PCA", "error", f"PCA_error_PC{comp_number}")
        # Stacked version
        plot_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA Error - PC{comp_number} Components and Explained Variance',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            fpath=os.path.join(ROOT_DIR, "PCA", "error", "stacked", f"PCA_diff_PC{comp_number}"),
            color_mapping=color_mapping,
            normalized=True,
            stacked=True
        )
        # Non-stacked version
        plot_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA Error - PC{comp_number} Components and Explained Variance (non-stacked)',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            fpath=os.path.join(ROOT_DIR, "PCA", "error", "nonstacked", f"PCA_diff_PC{comp_number}"),
            color_mapping=color_mapping,
            normalized=True,
            stacked=False
        )
    # Grid plots for PCA error components (stacked)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_list, index=bin_labels)
        ax = axes[comp_index]
        plot_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA Error - PC{comp_number} Components and Explained Variance',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            color_mapping=color_mapping,
            normalized=True,
            stacked=True,
            ax=ax
        )
    plt.tight_layout()
    save_dir = os.path.join(ROOT_DIR, "PCA", "error")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "_PCA_error_components_grid.pdf"), bbox_inches='tight')
    plt.close(fig)
    # Grid plots for PCA error components (non-stacked)
    fig_ns, axes_ns = plt.subplots(2, 2, figsize=(20, 12))
    axes_ns = axes_ns.flatten()
    for comp_index in range(n_components):
        comp_number = comp_index + 1
        pc_df = pd.DataFrame(pc_loadings[comp_index], columns=models_list, index=bin_labels)
        ax = axes_ns[comp_index]
        plot_area(
            pc_df,
            pc_explained[comp_index],
            title=f'PCA Error - PC{comp_number} Components and Explained Variance (non-stacked)',
            xlabel='Hurst Exponent Bin',
            ylabel=f'PC{comp_number} Loading',
            rmse_ylabel='Explained Variance (%)',
            color_mapping=color_mapping,
            normalized=True,
            stacked=False,
            ax=ax
        )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "_PCA_error_components_grid_nonstacked.pdf"), bbox_inches='tight')
    plt.close(fig_ns)
    logger.info("PCA error components plots saved.")

# =============================================================================
# Measure plots (with added non-stacked versions)
# =============================================================================

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

from scipy.stats import kendalltau
import numpy as np

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

def plot_measure_across_bins_generic(
    predictions, orig, measure_func, measure_name, transform_func, color_mapping, normalize=True
):
    """
    Generic function that computes a measure (using measure_func) for each bin.
    transform_func takes (target_vals, other_vals, true_vals, target) and returns
    the pair of arrays on which the measure is computed.
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
            true_vals = orig_arr[indices]
            for other in predictions:
                if other == target:
                    continue
                other_vals = np.array(predictions[other])[indices]
                t_target, t_other = transform_func(target_vals, other_vals, true_vals, target)
                val = measure_func(t_target, t_other)
                measure_data[other].append(val)
        if not bin_labels:
            continue
        df = pd.DataFrame(measure_data, index=bin_labels)
        final_title = f"{target} {measure_name} with Others"
        if normalize:
            df = df.div(df.sum(axis=1).replace(0, 1), axis=0)
            final_title += " (normalized)"
        rmse_primary = [0] * len(df.index)
        # Stacked version
        plot_area(
            df,
            rmse_primary,
            title=final_title,
            xlabel="Hurst Exponent Bin",
            ylabel=f"{'Normalized ' if normalize else ''}{measure_name}",
            rmse_ylabel=None,
            fpath=os.path.join(
                ROOT_DIR, "CorrMtx", target, "stacked", "normalized" if normalize else "original",
                f"{target}_{measure_name}"
            ),
            color_mapping=color_mapping,
            normalized=normalize,
            stacked=True
        )
        # Non-stacked version
        plot_area(
            df,
            rmse_primary,
            title=final_title + " (non-stacked)",
            xlabel="Hurst Exponent Bin",
            ylabel=f"{'Normalized ' if normalize else ''}{measure_name}",
            rmse_ylabel=None,
            fpath=os.path.join(
                ROOT_DIR, "CorrMtx", target, "nonstacked", "normalized" if normalize else "original",
                f"{target}_{measure_name}"
            ),
            color_mapping=color_mapping,
            normalized=normalize,
            stacked=False
        )

def plot_measure_across_bins_single(predictions, orig, measure_func, measure_name, color_mapping=None, normalize=True):
    transform = lambda t, o, true, target: (t, o)
    plot_measure_across_bins_generic(predictions, orig, measure_func, measure_name, transform, color_mapping, normalize)

def plot_measure_across_bins_single_error(
    predictions, orig, measure_func, measure_name, color_mapping=None, normalize=True
):
    transform = lambda t, o, true, target: (t - true, o - true)
    plot_measure_across_bins_generic(predictions, orig, measure_func, measure_name, transform, color_mapping, normalize)

def plot_measures_grid_generic(target, predictions, orig, method, color_mapping, normalize=True):
    measures = [
        ("Covariance", calc_cov),
        ("Pearson", calc_pearson),
        ("Spearman", calc_spearman),
        ("Mutual_Information", calc_mutual_info),
        ("Slope", calc_slope),
        ("Distance_Correlation", calc_distance_corr),
        ("Maximal_Information_Coefficient", calc_mic),
        ("Kendalls_tau", calc_kendall_tau),
        ("Blomqvists_beta", calc_blomqvist_beta),
    ]
    orig_arr = np.array(orig)
    bin_defs = get_custom_bin_defs(BIN_SIZE)
    fig, axes = plt.subplots(3, 3, figsize=(28, 18))
    axes = axes.flatten()

    if method == "predictions":
        transform = lambda t, o, true, target: (t, o)
    elif method == "errors":
        transform = lambda t, o, true, target: (t - true, o - true)
    else:
        raise ValueError(f"Unknown method: {method}")

    for idx, (measure_name, measure_func) in enumerate(measures):
        measure_data = {other: [] for other in predictions if other != target}
        bin_labels = []
        for (label, cond_fn) in bin_defs:
            indices = np.where(cond_fn(orig_arr))[0]
            if len(indices) == 0:
                continue
            bin_labels.append(label)
            target_vals = np.array(predictions[target])[indices]
            true_vals = orig_arr[indices]
            for other in predictions:
                if other == target:
                    continue
                other_vals = np.array(predictions[other])[indices]
                t_target, t_other = transform(target_vals, other_vals, true_vals, target)
                val = measure_func(t_target, t_other)
                measure_data[other].append(val)
        if not bin_labels:
            continue
        df = pd.DataFrame(measure_data, index=bin_labels)
        if normalize:
            df = df.div(df.sum(axis=1).replace(0, 1), axis=0)
        rmse_primary = [0] * len(df.index)
        method_title = {"predictions": "Predictions", "errors": "Errors"}[method]
        ax = axes[idx]
        plot_area(
            df,
            rmse_primary,
            title=f'{target} {measure_name} {method_title}' + (" (normalized)" if normalize else ""),
            xlabel="Hurst Exponent Bin",
            ylabel=f"{'Normalized ' if normalize else ''}{measure_name}",
            rmse_ylabel=None,
            color_mapping=color_mapping,
            normalized=normalize,
            stacked=True,
            ax=ax
        )
    suptitle = f'{target} Measures {method.capitalize()}' + (" (normalized)" if normalize else "")
    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_dir = os.path.join(ROOT_DIR, "CorrMtx", target, "stacked", "normalized" if normalize else "original")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"_{target}_measures_{method}_grid.pdf"), bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Plotted grid {suptitle}")

    # Now create a non-stacked grid version
    fig_ns, axes_ns = plt.subplots(3, 3, figsize=(28, 18))
    axes_ns = axes_ns.flatten()
    for idx, (measure_name, measure_func) in enumerate(measures):
        measure_data = {other: [] for other in predictions if other != target}
        bin_labels = []
        for (label, cond_fn) in bin_defs:
            indices = np.where(cond_fn(orig_arr))[0]
            if len(indices) == 0:
                continue
            bin_labels.append(label)
            target_vals = np.array(predictions[target])[indices]
            true_vals = orig_arr[indices]
            for other in predictions:
                if other == target:
                    continue
                other_vals = np.array(predictions[other])[indices]
                t_target, t_other = transform(target_vals, other_vals, true_vals, target)
                val = measure_func(t_target, t_other)
                measure_data[other].append(val)
        if not bin_labels:
            continue
        df = pd.DataFrame(measure_data, index=bin_labels)
        if normalize:
            df = df.div(df.sum(axis=1).replace(0, 1), axis=0)
        rmse_primary = [0] * len(df.index)
        ax = axes_ns[idx]
        plot_area(
            df,
            rmse_primary,
            title=f'{target} {measure_name} {method_title} (non-stacked)' + (" (normalized)" if normalize else ""),
            xlabel="Hurst Exponent Bin",
            ylabel=f"{'Normalized ' if normalize else ''}{measure_name}",
            rmse_ylabel=None,
            color_mapping=color_mapping,
            normalized=normalize,
            stacked=False,
            ax=ax
        )
    plt.tight_layout()
    save_dir = os.path.join(ROOT_DIR, "CorrMtx", target, "nonstacked", "normalized" if normalize else "original")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"_{target}_measures_{method}_grid_nonstacked.pdf"), bbox_inches='tight')
    plt.close(fig_ns)

def plot_measures_grid(predictions, orig, target, color_mapping, normalize=True):
    for method in ["predictions", "errors"]:
        plot_measures_grid_generic(target, predictions, orig, method, color_mapping, normalize)

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
    mse_dict = compute_mse(orig, predictions)
    logger.info(f"MSE values: {mse_dict}")
    mapping = get_color_mapping(list(predictions.keys()))

    for baseline in models.keys():
        pred = {name: pred for name, pred in predictions.items() if name != baseline}
        plot_scatter(
            pred,
            predictions[baseline],
            mapping,
            f"{baseline} vs others",
            xlabel=f"Hurst by {baseline}",
            ylabels={model: f"Hurst by {model}" for model in pred.keys()}
            
        )

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

    plot_scatter(predictions, orig, mapping, "true vs predicted Hurst")
    plot_stacked_area_LinReg(orig, predictions, baseline="LSTM", color_mapping=mapping)
    plot_stacked_area_SLSQP(orig, predictions, baseline="LSTM", color_mapping=mapping)

    # PCA plots
    plot_pca_components(orig, predictions, color_mapping=mapping)
    plot_pca_diff_components(orig, predictions, color_mapping=mapping)
    plot_pca_error_components(orig, predictions, color_mapping=mapping)

    # Measure plots (both non-normalized and normalized)
    for measure_name, measure_func in [
        ("Covariance", calc_cov),
        ("Pearson", calc_pearson),
        ("Spearman", calc_spearman),
        ("Mutual_Information", calc_mutual_info),
        ("Slope", calc_slope),
        ("Distance_Correlation", calc_distance_corr),
        ("Maximal_Information_Coefficient", calc_mic),
        ("Kendalls_tau", calc_kendall_tau),
        ("Blomqvists_beta", calc_blomqvist_beta),
    ]:
        # Raw predictions
        plot_measure_across_bins_single(
            predictions, orig, measure_func, measure_name, color_mapping=mapping, normalize=False
        )
        plot_measure_across_bins_single_error(
            predictions, orig, measure_func, measure_name, color_mapping=mapping, normalize=False
        )
        # Normalized versions
        plot_measure_across_bins_single(
            predictions, orig, measure_func, measure_name, color_mapping=mapping, normalize=True
        )
        plot_measure_across_bins_single_error(
            predictions, orig, measure_func, measure_name, color_mapping=mapping, normalize=True
        )

    # Grid plots for each target (raw and normalized)
    for target in predictions.keys():
        plot_measures_grid(predictions, orig, target, color_mapping=mapping, normalize=False)
        plot_measures_grid(predictions, orig, target, color_mapping=mapping, normalize=True)
    logger.info("Hurst exponent estimation process completed.")

if __name__ == "__main__":
    run_estimation(setup_models())
    logger.info("Process finished successfully.")

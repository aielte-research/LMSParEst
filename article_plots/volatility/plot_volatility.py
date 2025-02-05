from bokeh.palettes import Category10
import torch
import sys
sys.path.append('../')
sys.path.append('../../')
from article_plots.load_models import load_models, to_cuda
from metrics.plotters import general_plot

def plot_volatility(vol,dates, name, fname, window=252, stride=63, labels=None):
    r_over_s, variogram, higuchi, whittle, lstm, conv1d, lstm_non_std1, lstm_non_std2 = load_models()

    vol_tensor=torch.FloatTensor([vol]).unfold(1, window, stride)[0]
    measured_dates=[dates[x[int(len(x)/2)]] for x in torch.IntTensor([list(range(len(dates)))]).unfold(1, window, stride)[0]]

    lstm_estimates = [float(val[0])  for val in lstm(to_cuda(vol_tensor)).detach().cpu()]
    conv1d_estimates = [float(val[0])  for val in conv1d(to_cuda(vol_tensor)).detach().cpu()]
    lstm_non_std1_estimates = [float(val[0])  for val in lstm_non_std1(to_cuda(vol_tensor)).detach().cpu()]
    lstm_non_std2_estimates = [float(val[0])  for val in lstm_non_std2(to_cuda(vol_tensor)).detach().cpu()]

    if labels is None:
        labels = [f"{name} Log-Volatility","R/S Hurst","variogram Hurst","Higuchi Hurst","Whittle Hurst",r"$M_{conv}$ Hurst",r"$M_{LSTM}$ Hurst",r"$M_{LSTM}^*$ Hurst",r"$M_{LSTM}^*$ Hurst (std inference)"]

    general_plot({
        "Ys": [
            vol,
            [float(val) for val in r_over_s(vol_tensor)],
            [float(val) for val in variogram(vol_tensor)],
            [float(val) for val in higuchi(vol_tensor)],
            [float(val) for val in whittle(vol_tensor)],
            conv1d_estimates,
            lstm_estimates,
            lstm_non_std2_estimates,
            lstm_non_std1_estimates
        ],
        "Xs": [dates]+[measured_dates]*8,
        "xlabel": "Date",
        "ylabel": "Volatility and Hurst-estimate",
        "title": "",
        "fname": fname,
        "dirname": "volatility/plots",
        "markers": None,#[None,".",".",".",".",".","."],
        "legend": {
            "location": "top_left",
            "labels": labels
        },
        # "baselines":{
        #     "labels": [],
        #     "values": [0.5],
        #     "vertical": False,
        #     "colors": ["black"], # can be shorter than names
        #     "dashes": ["solid"] # can be shorter than namesself.colors
        # },
        "matplotlib": {
            "calc_xtics": False,
            # "width": 8.1,
            # "height": 4.725,
            "width": 6.875, #6.7,
            "height": 5,
            "style": "default"
        },
        "colors":  ["grey","red","red","red","red"]+[Category10[10][0],Category10[10][0],Category10[10][2],Category10[10][2]],
        #"colors":  ["grey"]+[Category10[10][i+2] for i in range(4)]+[Category10[10][0],Category10[10][1]],
        "dashes": ["solid","dotted","dashdot","dashed","solid","dashed","solid","solid","dashed"],
        "line45_color": None,
        "color_settings":{
            "bg_transparent": False
        }
    })
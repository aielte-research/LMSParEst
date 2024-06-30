from bokeh.palettes import Category10
import numpy as np
import torch
import yfinance as yf
import pandas as pd
import sys
sys.path.append('../')
from plotters import general_plot
sys.path.append('../../')
from load_models import load_models, to_cuda


data = yf.download("^GSPC", start="2000-01-01", end="2024-05-31")
print(data)
data['calc'] = data['Adj Close']
data['calc'] = np.log(data['calc'] / data['calc'].shift(1))
print(data)
data['calc'] = data['calc'].rolling(window=5).std()* np.sqrt(5)
print(data)
#data['calc'] = np.log(data['calc'])
#data['calc'] = data['calc'].diff()
data = data.dropna()
data = data.iloc[::5, :]
print(data)

vol=data['calc'].tolist()#[:2520]
dates=data['calc'].index.tolist()#[:2520]

r_over_s, variogram, higuchi, whittle, lstm, conv1d = load_models()

window=126
stride=21
# window=100
# stride=90

vol_tensor=torch.FloatTensor([vol]).unfold(1, window, stride)[0]
measured_dates=[dates[x[int(len(x)/2)]] for x in torch.IntTensor([list(range(len(dates)))]).unfold(1, window, stride)[0]]

lstm_estimates = [float(val[0])  for val in lstm(to_cuda(vol_tensor)).detach().cpu()]
conv1d_estimates = [float(val[0])  for val in conv1d(to_cuda(vol_tensor)).detach().cpu()]

gp_params={
    "Ys": [
        vol,
        [float(val) for val in r_over_s(vol_tensor)],
        [float(val) for val in variogram(vol_tensor)],
        [float(val) for val in higuchi(vol_tensor)],
        [float(val) for val in whittle(vol_tensor)],
        conv1d_estimates,
        lstm_estimates
    ],
    "Xs": [dates]+[measured_dates]*6,
    "xlabel": "Date",
    "ylabel": "Volatility and Hurst-estimate",
    "title": "",
    "fname": "volatility_SnP500_5day_no_overlaps",
    "dirname": "plots",
    "markers": None,#[None,".",".",".",".",".","."],
    "legend": {
        "location": "top_left",
        "labels": ["S&P 500 daily Log-Volatility","R/S Hurst","variogram Hurst","Higuchi Hurst","Whittle Hurst","1D CNN Hurst","LSTM Hurst"]
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
        "width": 8.1,
        "height": 4.725,
        "style": "default"
    },
    "colors":  ["grey","red","red","red","red"]+[Category10[10][1],Category10[10][0]],
    #"colors":  ["grey"]+[Category10[10][i+2] for i in range(4)]+[Category10[10][0],Category10[10][1]],
    "dashes": ["solid","dotted","dashdot","dashed","solid","solid","solid"],
    "line45_color": None,
    "color_settings":{
        "bg_transparent": False
    }
}
general_plot(gp_params)
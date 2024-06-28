from bokeh.palettes import Category10
import numpy as np
import torch
import yfinance as yf

import sys
from plotters import general_plot
sys.path.append('../')
from models.baselines.R_over_S import Model as R_over_S
from models.baselines.variogram import Model as Variogram
from models.baselines.higuchi import Model as Higuchi
from models.baselines.whittle import Model as Whittle
from models.LSTM import Model as LSTM
from models.base import Model as Conv1d

data = yf.download("^VIX", start="2000-01-01", end="2023-12-31")
print(data)
data['calc'] = data['Adj Close']
data['calc'] = 0.01*data['calc'].dropna()
#data['calc'] = np.log(data['calc']).dropna()

# data = yf.download("^GSPC", period="2y", interval="60m")
# data = yf.download("^GSPC", start="1990-01-01", end="2024-05-31")
# print(data)
# data['calc'] = data['Adj Close']
# data['calc'] = np.log(data['calc'] / data['calc'].shift(1)).dropna()
# print(data)
# data['calc'] = data['calc'].rolling(window=21).std()*np.sqrt(21) #* np.sqrt(252)
# #data['calc'] = np.log(data['calc'])
# log_volatility = data['calc']#.diff()
# data = data.dropna()

vol=data['calc'].tolist()#[:2520]
dates=data['calc'].index.tolist()#[:2520]

diff=True
r_over_s=R_over_S({'diff':diff},None)
variogram=Variogram({'diff':diff},None)
higuchi=Higuchi({'diff':diff},None)
whittle=Whittle({'diff':diff},None)

def to_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var

model_params={
    "diff": True,
    "standardize": True,
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

state_dict_path="../model_checkpoints/fBm/fBm_Hurst_LSTM_finetune_until_n-12800.pt"
lstm = to_cuda(LSTM(model_params, state_dict_path))
lstm.eval()

model_params={
    "diff": True,
    "standardize": True,
    "invariant_nu_layer": None,
    "additive_nu_layer": None,
    "init_hom": None,
    "rs": None,
    "embedding": {
        "bias": True,
        "channels": (1, 64, 64, 128, 128, 128, 128),
        "kernel_sizes": 4,
        "activation":{"name": "PReLU"}
    },
    "vec_hom": None,
    "avg_base": None,
    "nu_estimator": False,
    "simple_nu_estimator": False,
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

state_dict_path="../model_checkpoints/fBm/fBm_Hurst_conv1D_finetune_until_n-12800.pt"
conv1d = to_cuda(Conv1d(model_params, state_dict_path))
conv1d.eval()

window=252
stride=126
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
    "fname": "SnP500_volatility",
    "dirname": "plots",
    "markers": None,#[None,".",".",".",".",".","."],
    "legend": {
        "location": "top_left",
        "labels": ["CBOE Volatility Index","R/S Hurst","variogram Hurst","Higuchi Hurst","Whittle Hurst","1D CNN Hurst","LSTM Hurst"]
    },
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
from bokeh.palettes import Category10
import numpy as np
import torch
import yfinance as yf
import pandas as pd
from load_models import load_models, to_cuda
import sys
sys.path.append('../')
from plotters import general_plot

# data = pd.read_csv('SnP500-15m.csv', sep=';', names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

# print(data)
# data['calc'] = data['Close']
# data['calc'] = np.log(data['calc'] / data['calc'].shift(1))
# print(data)
# data = data.groupby('Date').std().reset_index()
# data['calc'] = data['calc'] *np.sqrt(252)
# # print(data)


df = pd.read_csv('SnP500-15m.csv', sep=';', names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

print(df)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],format="%d/%m/%Y %H:%M:%S")
df['Date'] = pd.to_datetime(df['Date'],format="%d/%m/%Y")
df = df.set_index('Datetime')
df = df.drop(columns=['Time'])
# df = df.resample('1H').agg({
#     'Date': 'first',
#     'Open': 'first',
#     'High': 'max',
#     'Low': 'min',
#     'Close': 'last',
#     'Volume': 'sum',
# }).dropna()

print(df)
df['calc'] = df['Close']
df['calc'] = np.log(df['calc'] / df['calc'].shift(1))
df=df.dropna()
print(df)
df = df.resample('1D').agg({
    'Date': 'first',
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum',
    'calc': 'std'
}).dropna()
print(df)
# df = df.groupby('Date').std().reset_index()
# print(df)
df = df.set_index('Date')
df = df.sort_values(by=['Date'])
df['calc'] = df['calc'] *np.sqrt(24*4)
print(df)
data = df

vol=data['calc'].tolist()#[:2520]
dates=data['calc'].index.tolist()#[:2520]

r_over_s, variogram, higuchi, whittle, lstm, conv1d = load_models()

window=252
stride=63
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
    "fname": "volatility_SnP500_hour-day",
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
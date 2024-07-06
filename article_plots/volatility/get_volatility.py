import numpy as np
import yfinance as yf

def get_volatility(key="BTC-USD",overlap=False,start="1990-01-01", end="2024-05-31"):
    data = yf.download(key, start=start, end=end)
    print(data)
    data['calc'] = data['Adj Close']
    data['calc'] = np.log(data['calc'] / data['calc'].shift(1)).dropna()
    print(data)
    if overlap:
        data['calc'] = data['calc'].rolling(window=21).std() * np.sqrt(252)
        data = data.dropna()
    else:
        data['calc'] = data['calc'].rolling(window=5).std()* np.sqrt(5)
        data = data.dropna()
        data = data.iloc[::5, :]

    vol=data['calc'].tolist()
    dates=data['calc'].index.tolist()

    return vol, dates
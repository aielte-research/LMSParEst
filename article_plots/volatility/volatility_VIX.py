import yfinance as yf
from plot_volatility import plot_volatility

data = yf.download("^VIX", start="2000-01-01", end="2023-12-31")
print(data)
data['calc'] = data['Adj Close']
#data['calc'] = data['High']
data['calc'] = 0.01*data['calc'].dropna()
#data['calc'] = np.log(data['calc']).dropna()

vol=data['calc'].tolist()#[:2520]
dates=data['calc'].index.tolist()#[:2520]

plot_volatility(
    vol, dates, "CBOE Volatility Index", f"volatility_VIX", window=252, stride=63,
    labels = ["CBOE Volatility Index","R/S Hurst","variogram Hurst","Higuchi Hurst","Whittle Hurst","1D CNN Hurst","LSTM Hurst"]            
)
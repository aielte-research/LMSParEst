from alpaca.data.historical import StockHistoricalDataClient

# No keys required for crypto data
client = StockHistoricalDataClient()

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Creating request object
request_params = StockBarsRequest(
  symbol_or_symbols=["^GSPC"],
  timeframe=TimeFrame.Hour,
  start="2000-01-01",
  end="2024-06-30"
)

# Retrieve daily bars for Bitcoin in a DataFrame and printing it
btc_bars = client.get_stock_bars(request_params)

# Convert to dataframe
print(btc_bars.df)
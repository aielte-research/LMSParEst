from get_volatility import get_volatility
from plot_volatility import plot_volatility

for overlap in [True, False]:
    vol, dates = get_volatility(key="DIS", overlap=overlap, start="1960-01-01", end="2024-06-30")

    name="Disney"
    if overlap:
        plot_volatility(vol, dates, name, f"volatility_Disney_monthly_overlap", window=252, stride=126)
    else:
        plot_volatility(vol, dates, name, f"volatility_Disney_weekly_no_overlap", window=100, stride=50)
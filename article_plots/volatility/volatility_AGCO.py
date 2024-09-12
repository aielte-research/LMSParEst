from get_volatility import get_volatility
from plot_volatility import plot_volatility

for overlap in [True, False]:
    vol, dates = get_volatility(key="AGCO", overlap=overlap, start="2000-01-01", end="2024-06-30")

    name="AGCO"
    if overlap:
        plot_volatility(vol, dates, name, f"volatility_AGCO_monthly_overlap", window=252, stride=63)
    else:
        plot_volatility(vol, dates, name, f"volatility_AGCO_weekly_no_overlap", window=100, stride=25)
from get_volatility import get_volatility
from plot_volatility import plot_volatility
import numpy as np

for overlap in [True, False]:
    vol, dates = get_volatility(key="GME", overlap=overlap, start="2007-01-01", end="2024-06-30")

    name="GameStop"
    if overlap:
        vol = list(np.asarray(vol) * (np.sqrt(21)/np.sqrt(252)))
        plot_volatility(vol, dates, name, f"volatility_{name}_monthly_overlap", window=252, stride=63)
    else:
        vol = list(np.asarray(vol) * (1/np.sqrt(5)))
        plot_volatility(vol, dates, name, f"volatility_{name}_weekly_no_overlap", window=100, stride=25)
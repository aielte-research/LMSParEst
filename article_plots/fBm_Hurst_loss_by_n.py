from bokeh.palettes import Category10
import sys
sys.path.append('../')
from metrics.plotters import general_plot

## Comparing different model performances by sequence length
# PEFBM-4788 - PEFBM-4860

general_plot({
    "Ys": [
        [0.0276449, 0.0189365, 0.0138884, 0.0108062, 0.00862036, 0.00673888, 0.00557211, 0.00469968],
        [0.00930144, 0.00505078, 0.00291788, 0.00174502, 0.00109185, 0.000723842, 0.000502304, 0.000365104],
        [0.0105847, 0.00420909, 0.0019889, 0.00104599, 0.000593053, 0.00036006, 0.000228544, 0.000155049],
        [0.0043292, 0.00199614, 0.00100069, 0.000540158, 0.000324259, 0.000225005, 0.00017898, 0.000157218],
        [0.004266, 0.00199481, 0.000959342, 0.000475629, 0.000240011, 0.000122124, 0.000062808, 0.0000332803],
        [0.00407246, 0.0018894, 0.000912424, 0.000453153, 0.000224341, 0.000113915, 0.000057925, 0.0000296729],
    ],
    "Xs": [100,200,400,800,1600,3200,6400,12800],
    "xlabel": "Sequence Length (n)",
    "xscale": "log",
    "ylabel": "MSE Loss",
    "yscale": "log",
    "title": "",
    "fname": "fBm_Hurst_loss_by_n_loglog",
    "dirname": "./plots",
    "markers": None,
    "legend": {
        "location": "bottom_left",
        "labels": ["R/S","variogram","Higuchi","Whittle","1D CNN","LSTM"]#,"Our model with transformer encoder"]
    },
    "matplotlib": {
        "calc_xtics": False,
        "width": 6.6,
        "height": 3.85,
        "style": "default"
    },
    "colors": ["red","red","red","red"]+[Category10[10][1],Category10[10][0]],
    "dashes": ["dotted","dashdot","dashed","solid","solid","solid"],
    "line45_color": None,
    "color_settings":{
        "bg_transparent": False
    }
})
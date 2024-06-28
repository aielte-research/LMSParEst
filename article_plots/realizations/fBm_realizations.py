import sys
sys.path.append('..')
from plotters import general_grid_plot
sys.path.append('../..')
from process_generators.fbm_gen import gen as fbm_gen

plot_params={
    "fname": "fBm_realizations",
    "markers": None,
    "dirname": "../plots",
    "matplotlib":{
        "calc_xtics": False,
        "width": 10,
        "height": 2.5,
    },
    "color_settings":{
        "bg_transparent": False
    }
}

grid_params=[]

#for H in [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]:
for H in [0.01, 0.25, 0.5, 0.75, 0.99]:
    params = plot_params.copy()
    params["Ys"] = list(fbm_gen(hurst = H, n = 256))
    params["title"] = f"H = {H}"
    grid_params.append(params)

general_grid_plot(grid_params, width=1)
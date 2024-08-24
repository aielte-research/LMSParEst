import sys
sys.path.append('..')
sys.path.append('../..')
from metrics.plotters import general_grid_plot
from process_generators.fou_gen import gen as fou_gen

plot_params={
    "fname": "fOU_realizations",
    "markers": None,
    "dirname": "plots",
    "matplotlib":{
        "calc_xtics": False,
        "width": 8,
        "height": 2,
    },
    "color_settings":{
        "bg_transparent": False
    }
}

grid_params=[]

for H in [0.01, 0.25, 0.5, 0.75, 0.99]:
    for alpha in [0, 0.05, 0.5, 5]:
        params = plot_params.copy()
        params["Ys"] = list(fou_gen(hurst = H, alpha=alpha, n = 256, dt=1))
        params["title"] = f"H = {H}, α = {alpha}"
        grid_params.append(params)

general_grid_plot(grid_params, width=4)
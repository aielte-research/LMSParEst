import sys
sys.path.append('..')
from plotters import general_grid_plot
sys.path.append('../..')
from process_generators.arfima_gen import gen as arfima_gen

plot_params={
    "fname": "ARFIMA_realizations",
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

#for d in [-0.5, -0.49, -0.25, 0, 0.25, 0.49, 0.5]:
for d in [-0.49, -0.25, 0, 0.25, 0.49]:
    params = plot_params.copy()
    params["Ys"] = list(arfima_gen(d = d, n = 256))
    params["title"] = f"d = {d}"
    grid_params.append(params)

general_grid_plot(grid_params, width=1)
import numpy as np
from metrics.plotters import GeneralPlotter

def calc_mse(values):
    if len(values) == 0:
        return 0
    return (1/len(values)) * sum([val**2 for val in values])

class Mse():
    def __init__(self, params, neptune_experiment):
        default_params={
            "plot_limit": 50000,
            "measure_interval": 0.25,
            "steps": 2000,
            "title": ""
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        for k, v in params.items():
            setattr(self, k, v)

        self.neptune_experiment = neptune_experiment

    def __call__(self, data):
        goals=[]
        results=[]
        for pair in data:
            goal=pair[0][-self.plot_limit:]
            result=pair[1][-self.plot_limit:]
            permut = np.argsort(goal)
            goals.append(np.array(goal)[permut])
            results.append(np.array(result)[permut])

        range_a = min([cfg["data_params"][self.title]["a"] for cfg in self.serialized_cfg])
        range_b = max([cfg["data_params"][self.title]["b"] for cfg in self.serialized_cfg])

        mse_lst = []
        x_range = np.linspace(range_a, range_b, num=self.steps)
        for X, Y in zip(goals, results):
            mse = []
            
            zero_centered = [i-t for t,i in zip(X,Y)]

            for xval in  x_range:
                values=zero_centered[np.searchsorted(X, xval-self.measure_interval):np.searchsorted(X, xval+self.measure_interval,side='right')]

                mse.append(calc_mse(values))

            mse_lst.append(mse)
        
        params={
            "Ys": mse_lst,
            "x": x_range,
            "xlabel": self.title,
            "ylabel": "MSE",
            "title": "FBM MSE (steps: {}, interval:+-{})".format(self.steps,self.measure_interval),
            "fname": "{}_mse_{}".format(self.title.replace(" ","_"), self.serialized_cfg[0]["experiment_name_base"]),
            "dirname": "./plots",
            "markers": None,
            "legend":{
                "location": "top_left", # "top_left", "top_right", "bottom_left", "bottom_right", None
                "labels": [cfg["experiment_name"] for cfg in self.serialized_cfg]
            },
            "matplotlib":{ # for png and svg
                "calc_xtics": False
            }
        }

        GeneralPlotter(params, neptune_experiment=self.neptune_experiment).export_all()
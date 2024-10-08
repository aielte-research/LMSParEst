import numpy as np
from metrics.plotters import general_plot

def calc_dev(goals, results, range_a, range_b, steps, measure_interval):
    deviations_lst = []
    biases_lst = []
    deviation_aucs = []
    bias_aucs = []
    x_range = list(np.linspace(range_a, range_b, num=steps))

    for X_, Y_ in zip(goals, results):
        permut = np.argsort(X_)
        X = np.array(X_)[permut]
        Y = np.array(Y_)[permut]
        deviations=[]
        biases=[]
        deviation_aucs.append(0)
        bias_aucs.append(0)
        
        zero_centered=[i-t for t,i in zip(X,Y)]

        for xval in  x_range:
            values=zero_centered[np.searchsorted(X, xval-measure_interval):np.searchsorted(X, xval+measure_interval,side='right')]

            dev=np.std(values)
            deviations.append(dev)
            deviation_aucs[-1]+=abs(dev)

            if len(values)==0:
                avg = 0
            else:
                avg = sum(values)/len(values)
            biases.append(avg)
            bias_aucs[-1]+=abs(avg)
        deviation_aucs[-1]*=x_range[1]-x_range[0]
        bias_aucs[-1]*=x_range[1]-x_range[0]

        deviations_lst.append(deviations)
        biases_lst.append(biases)
    return x_range, deviations_lst, biases_lst, deviation_aucs, bias_aucs

class Deviation():
    def __init__(self, params, neptune_experiment):
        default_params={
            "plot_limit": 50000,
            "measure_interval": 0.25,
            "steps": 2000,
            "title": "hurst"
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        for k, v in params.items():
            setattr(self, k, v)

        self.neptune_experiment=neptune_experiment

    def __call__(self, data):
        goals=[]
        results=[]
        for pair in data:
            goals.append(pair[0][-self.plot_limit:])
            results.append(pair[1][-self.plot_limit:])

        range_a = min([cfg["data_params"][self.title]["a"] for cfg in self.serialized_cfg])
        range_b = max([cfg["data_params"][self.title]["b"] for cfg in self.serialized_cfg])

        x_range, deviations_lst, biases_lst, deviation_aucs, bias_aucs = calc_dev(goals, results, range_a, range_b, self.steps, self.measure_interval)

        legend_labels = [f'{cfg["experiment_name"]} (AUC:{auc})' for cfg,auc in zip(self.serialized_cfg,deviation_aucs)]
        general_plot({
            "Ys": deviations_lst,
            "Xs": x_range,
            "xlabel": self.title,
            "ylabel": "deviation",
            "title": "Deviations (steps: {}, interval:+-{})".format(self.steps,self.measure_interval),
            "fname": "{}_deviations_{}".format(self.title.replace(" ","_"), self.serialized_cfg[0]["experiment_name_base"]),
            "dirname": "./plots",
            "markers": None,
            "legend":{
                "location": "top_left", # "top_left", "top_right", "bottom_left", "bottom_right", None
                "labels": legend_labels
            },
            "matplotlib":{ # for png and svg
                "calc_xtics": False
            },
            "neptune_experiment":self.neptune_experiment
        })

        legend_labels = [f'{cfg["experiment_name"]} (AUC:{auc})' for cfg,auc in zip(self.serialized_cfg,bias_aucs)]        
        general_plot({
            "Ys": biases_lst,
            "Xs": x_range,
            "xlabel": self.title,
            "ylabel": "bias",
            "title":  "Biases (steps: {}, interval:+-{})".format(self.steps,self.measure_interval),
            "fname": "{}_biases_{}".format(self.title.replace(" ","_"), self.serialized_cfg[0]["experiment_name_base"]),
            "dirname": "./plots",
            "markers": None,
            "legend":{
                "location": "top_right", # "top_left", "top_right", "bottom_left", "bottom_right", None
                "labels": legend_labels
            },
            "matplotlib":{ # for png and svg
                "calc_xtics": False
            },
            "neptune_experiment":self.neptune_experiment
        })
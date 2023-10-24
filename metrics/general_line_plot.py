from metrics.plotters import GeneralPlotter

class General_line_plot():
    def __init__(self, params, neptune_experiment):
        default_params={
            "xlabel": "epoch",
            "ylabel": "loss",
            "title": "Losses",
            "dirname": "./plots",
            "fname": "{}_losses".format(params['serialized_cfg'][0]["experiment_name_base"]),
            "legend_location": "top_right",
            "x_range":[]
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        for k, v in params.items():
            setattr(self, k, v)

        self.neptune_experiment=neptune_experiment

    def __call__(self, data):
        params={
            "Ys": data,
            "x": self.x_range,
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "title": self.title,
            "fname": self.fname,
            "dirname": "./plots",
            "legend":{
                "location": self.legend_location, # "top_left", "top_right", "bottom_left", "bottom_right", None
                "labels": [cfg["experiment_name"] for cfg in self.serialized_cfg]
            }
        }

        GeneralPlotter(params, neptune_experiment=self.neptune_experiment).export_all()
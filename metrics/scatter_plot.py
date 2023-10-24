from metrics.plotters import ScatterPlotter

class Scatter_plot():
    def __init__(self, params, neptune_experiment):
        default_params={
            "title": "scatter",
            "plot_limit": 5000,
            "line45_color": "red",
            "xlabel": "truth",
            "ylabel": "inferred",
            "opacity": 0.9,
            "legend_location": "bottom_right",
            "circle_size": 10,
            "x_jitter": 0,
            "heatmap": True
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        for k, v in params.items():
            setattr(self, k, v)

        self.neptune_experiment=neptune_experiment

    def __call__(self, data):
        goals=[goal[0][-self.plot_limit:] for goal in data]
        results=[result[1][-self.plot_limit:] for result in data]

        # for seq_idx in range(len(results)):
        #     for i in range(len(results[seq_idx])):
        #         if math.isnan(results[seq_idx][i]):
        #             results[seq_idx][i]="NaN"
        
        # print (results[0][:100])

        params={
            "Xs": goals,
            "Ys": results,
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "title": self.title,
            "fname": "{}_scatter_{}.html".format(self.title.replace(" ","_"), self.serialized_cfg[0]["experiment_name_base"]),
            "dirname": "./plots",
            "circle_size": self.circle_size,
            "x_jitter": self.x_jitter,
            "opacity": self.opacity,
            "heatmap": self.heatmap,
            "line45_color": self.line45_color,
            "legend":{
                "labels": [cfg["experiment_name"] for cfg in self.serialized_cfg],
                "location": self.legend_location, # "top_left", "top_right", "bottom_left", "bottom_right", None
            },
            "matplotlib":{ # for png and svg
                "width": 12.5,
                "height": 15,
            } 
        }

        ScatterPlotter(params, neptune_experiment=self.neptune_experiment).export_all()

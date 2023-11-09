import os
import json
from helper_functions import my_open

class Export_results():
    def __init__(self, params, neptune_experiment):
        self.fpath = params.get('fpath')
        self.export_targets = params.get('export_targets',False)
        self.prec = params.get('precision',17)
        self.neptune_experiment=neptune_experiment

    def __call__(self, data):
        (targets,),(results,)=zip(*data)

        extension = os.path.splitext(self.fpath)[1].strip(".")
        if extension == 'csv':
            with my_open(self.fpath, "w") as f:
                if self.export_targets:
                    f.write(f"result,target,diff,SE\n")
                    f.write("\n".join([f"{y:.{self.prec}f},{x:.{self.prec}f},{y-x:.{self.prec}f},{(y-x)**2:.{self.prec}f}" for x,y in zip(targets,results)]))
                else:
                    f.write(f"result\n")
                    f.write("\n".join([f"{y:.{self.prec}f}" for y in results]))
        elif extension == 'tsv':        
            with my_open(self.fpath, "w") as f:
                if self.export_targets:
                    f.write(f"result\ttarget\tdiff\tSE\n")
                    f.write("\n".join([f"{y:.{self.prec}f}\t{x:.{self.prec}f}\t{y-x:.{self.prec}f}\t{(y-x)**2:.{self.prec}f}" for x,y in zip(targets,results)]))
                else:
                    f.write(f"result\n")
                    f.write("\n".join([f"{y:.{self.prec}f}" for y in results]))
        elif extension == 'json':
            with my_open(self.fpath, "w") as f:
                if self.export_targets:
                    res={
                        "results":results,
                        "targets":targets,
                        "diff":[y-x for x,y in zip(targets,results)],
                        "SE":[(y-x)**2 for x,y in zip(targets,results)]
                    }
                else:
                    res={
                        "results":results
                    }                
                f.write(json.dumps(json.loads(json.dumps(res), parse_float=lambda x: round(float(x), self.prec)), indent=4))
        else:
            raise ValueError(f"Config extension unrecognized: '.{extension}'. Must be '.csv', '.tsv' or '.json'!")
        
        if self.neptune_experiment is not None:
            self.neptune_experiment[f"{extension}/{self.fpath}"].upload(self.fpath)
            self.neptune_experiment.sync() 

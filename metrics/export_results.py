import os
import json
from helper_functions import my_open

class Export_results():
    def __init__(self, params, neptune_experiment):
        self.fpath = params.get('fpath')
        self.export_targets = params.get('export_targets',False)
        self.neptune_experiment=neptune_experiment

    def __call__(self, data):
        (targets,),(results,)=zip(*data)

        extension = os.path.splitext(self.fpath)[1].strip(".")
        if extension == 'csv':
            with my_open(self.fpath, "w") as f:
                if self.export_targets:
                    f.write(f"result,target,diff,SE\n")
                    f.write("\n".join([f"{y},{x},{y-x},{(y-x)**2}" for x,y in zip(targets,results)]))
                else:
                    f.write("\n".join([str(y) for y in results]))
        elif extension == 'tsv':        
            with my_open(self.fpath, "w") as f:
                if self.export_targets:
                    f.write(f"result\ttarget\tdiff\tSE\n")
                    f.write("\n".join([f"{y}\t{x}\t{y-x}\t{(y-x)**2}" for x,y in zip(targets,results)]))
                else:
                    f.write("\n".join([str(y) for y in results]))
        elif extension == 'json':
            with my_open(self.fpath, "w") as f:
                res={
                    "results":results,
                    "targets":targets,
                    "diff":[y-x for x,y in zip(targets,results)],
                    "SE":[(y-x)**2 for x,y in zip(targets,results)]
                }
                f.write(json.dumps(res, indent=4))
        else:
            raise ValueError(f"Config extension unrecognized: '.{extension}'. Must be '.csv', '.tsv' or '.json'!")
        
        if self.neptune_experiment is not None:
            self.neptune_experiment[f"{extension}/{self.fpath}"].upload(self.fpath)
            self.neptune_experiment.sync() 

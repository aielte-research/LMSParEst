from models.base import BaseRegressor as Base
from metrics.plotters import spectrum_plot

class GetSpectrum(Base):
    def forward(self, x):
        if self.diff:
            x = x[:, :-1] - x[:, 1:]

        size_cond = (self.avg_base is not None) and (x.size(1) > self.avg_base)
        if size_cond:
            n_mult = int(x.size(1) / self.avg_base)
            x = x.view(x.size(0) * n_mult, -1)

        if self.standardize:            
            m = x.mean(dim = 1, keepdim = True)
            s = x.std(dim = 1, keepdim = True)
            x = (x - m) / s

        x = x.unsqueeze(dim = 1)
        # print(x.size())
        x = self.emb(x)
        
        return x

class Batch_spectrogram_plotter():
    def __init__(self, params, neptune_experiment):
        if "param_idx" in params:
            params["real_param_idx"]=params["param_idx"]
            params["inferred_param_idx"]=params["param_idx"]
        default_params={
            "limit": 10,
            "real_param_idx": 0,
            "inferred_param_idx": 0,
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        self.__dict__ = params

        self.spectrum_model = GetSpectrum(self.serialized_cfg[self.exp_idx]["model_params"])

        if self.train:
            self.batch_size = self.train_batch_size        
        else:
            self.batch_size = self.val_batch_size

    def __call__(self, true_params, inferred_params, batch_input):
        if self.batch_size * self.batch_count <= self.limit:
            specs = self.spectrum_model(batch_input.cpu())
            for idx, (spec, param) in enumerate(zip(specs,true_params.cpu().numpy())):
                spectrum_plot({
                    "spectrum": spec,
                    "title": f"param: {param[0]}",
                    "fname": f"spectrogram_{idx}",
                    "dirname": "./plots",
                    "neptune_experiment":self.neptune_experiment
                })
            #input()
        else:
            return []
class Batch_saver():
    def __init__(self, params, neptune_experiment):
        if "param_idx" in params:
            params["real_param_idx"]=params["param_idx"]
            params["inferred_param_idx"]=params["param_idx"]
        default_params={
            "start_count": 0,
            "real_param_idx": 0,
            "inferred_param_idx": 0,
        }
        for key in default_params:
            params[key] = params.get(key, default_params[key])

        self.__dict__ = params

        if self.train:
            self.batch_size = self.train_batch_size        
        else:
            self.batch_size = self.val_batch_size

    def __call__(self, true_params, inferred_params, _):
        to_be_saved = self.batch_size * self.batch_count >= self.start_count

        if to_be_saved:
            if true_params is None:
                return [[None, y[self.inferred_param_idx].item()] for y in inferred_params]
            else:
                return [[x[self.real_param_idx].item(), y[self.inferred_param_idx].item()] for x, y in zip(true_params, inferred_params)]
        else:
            return []
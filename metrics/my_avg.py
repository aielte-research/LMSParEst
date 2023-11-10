import os
class My_avg():
    def __init__(self, params, neptune_experiment):
        self.__dict__ = params
        self.neptune_experiment = neptune_experiment

    def __call__(self, seq):
        a = sum(seq) / len(seq)
        
        self.neptune_experiment[os.path.join('metrics', self.message)].log(a)

        return [a]
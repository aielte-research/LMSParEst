import functools as ft

def flatten(seq):
    return ft.reduce(lambda a, b: a + b, seq)

class Experiment_saver():
    def __init__(self, params, neptune_experiment):
        self.__dict__ = params

    def __call__(self, exp_res):
        try: 
            len(exp_res)
            len(exp_res[0])
        except:
            return [exp_res]                  
        
        ret = []
        for i in range(len(exp_res[0])):
            ret.append(flatten([a[i] for a in exp_res]))

        return [ret]

import functools as ft
import numpy as np

# class Exp_saver_data_class(np.ndarray):
#     def __new__(cls, input_array, info=None):
#         # Input array is an already formed ndarray instance
#         # We first cast to be our class type
#         obj = np.asarray(input_array).view(cls)
#         return obj
#     def __add__(self, x):
#         return np.concatenate((self, x), axis=1)
#     def __iadd__(self, x):
#         self=Exp_saver_data_class(np.concatenate((self, x), axis=1))
#         return self

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

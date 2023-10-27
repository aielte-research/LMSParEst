import torch
import importlib

from helper_functions import string_import

#TSPairs is a class from which we initiate objects that can be given to the Pytorch dataloader
#such an object has fixed epoch length paramater which determines
#how many series will be loaded by the dataloader in an epoch
#it also has other parameters(the ones ts_gen takes)
#these parameters can either be fixed or take a random value every time
#we declare fixed params simply by giving them value: e.g. n = 1500
#and random params by giving the key the '_gen' suffix and giving the generator itself
#e.g. hurst_gen = lamba: n_gen = lambda: random.randrange(3000, 10000)
#moreover if we dont want to infer a random value we can put its name to the no_inference list
class TSPairs(torch.utils.data.dataset.Dataset):
  
    def __init__(self, epoch_length, ts_gen, inference = [], extra_params = [],
                       const_in_batch = [], random_freq = 64, extra_batch_size = 1, **params):
        #the init method. this method will only be called once at the beginning
        #of the training session. it returns the actual object the dataloader can work with

        self.epoch_length = epoch_length
        self.ts_gen = ts_gen
        self.inference = inference
        self.extra_params = extra_params
        self.const_in_batch = const_in_batch
        self.random_freq = random_freq
        self.extra_batch_size = extra_batch_size

        #we seperate the fixed params from the generators
        self.fixed_params = {k: v for k, v in params.items() if not callable(v)}
        if extra_batch_size > 1:
            self.fixed_params['extra_batch_size'] = extra_batch_size
        self.param_gen = {k: v for k, v in params.items()
                          if (callable(v) and k not in const_in_batch)}

        self.const_batch_param_gen = {k: v for k, v in params.items()
                                      if (callable(v) and k in const_in_batch)}

        self.const_batch_list = [{k: v() for k, v in self.const_batch_param_gen.items()}
                                 for _ in range(int(self.epoch_length / self.random_freq) + 1)]
    def change_gen(self, new_gen):
        only_key = [k for k in self.param_gen.keys()][0]
        self.param_gen[only_key] = new_gen

    def __getitem__(self, idx):

        #this method is needed by the dataloader
        #it will be called several million times during the training session
        #each call will return a random series and an appropriate label

        #get random values from the generators
        rand_params = {k: v() for k, v in self.param_gen.items()}

        
        if idx == 0:
            self.const_batch_list = [{k: v() for k, v in self.const_batch_param_gen.items()}
                                     for _ in range(int(self.epoch_length / self.random_freq) + 1)]

        const_batch_params = self.const_batch_list[int(idx / self.random_freq)]
        rand_params.update(const_batch_params)
        all_params = dict(**self.fixed_params, **rand_params)

        label_list = ([all_params[k] for k in self.inference]
                    + [all_params[k] for k in self.extra_params])

        label = torch.FloatTensor(label_list)

        if self.extra_batch_size > 1:
            label = label.repeat(self.extra_batch_size, 1)

        series = self.ts_gen(**all_params)
        series = torch.FloatTensor(series)


        return series, label

    def __len__(self):
        return self.epoch_length


def import_param(k, v):

    if isinstance(v, dict) and ('gen' in v.keys()):

        gen = string_import(v['gen'])
        new_v = {a: b for a, b in v.items() if a != 'gen'}
        res = lambda: gen(**new_v)
    
    else:
        res = v
        
    return res

class Database():
    def __init__(self, config, train_freq, val_freq):
         #first we import our generator

        gen_module = importlib.import_module('process_generators.' + config['ts_type'] + '_gen')
        generator = gen_module.gen
        new_config = {k: import_param(k, v) for k, v in config.items()
                                      if k != 'ts_type'}
        train_pairs = TSPairs(**new_config, random_freq = train_freq,
                                                           ts_gen = generator)
        val_pairs = TSPairs(**new_config, random_freq = train_freq,
                                                           ts_gen = generator)
        self.database = (train_pairs, val_pairs)


'''
cfg = {
        'inference': ('hurst'),
        'epoch_length': 10000, 
        'ts_type': 'fou',
        'series_length': 1500,
        'hurst': {'gen': 'random.uniform', 'a': 0, 'b': 1},
        'time_scale': 1,
        'sigma': 0.3,
        'alpha': -0.1,
        'initial_value': 0
        }

Proba = Database(cfg, 64, 64)
print(Proba.database[0][0][1].size())

'''
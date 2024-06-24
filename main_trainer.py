import os, importlib
import argparse
import torch

import copy
import cfg_parser
import numpy as np
import random
import time

import yaml
from torch.utils import data
from helper_functions import pretty_time, string_import

import neptune

from tqdm import tqdm
import math

from unittest.mock import MagicMock

#------------------------------------------------------------------------------
#                          ABSTRACT LEVEL
#------------------------------------------------------------------------------

#first we create the abstract class MetaTrainer 
#it's a high level class that isn't necessarily DL specific
#but later we will subclass it for our DL specific purposes (ExperimentTrainer, EpochTrainer, BatchTrainer)

#we can think of it as a dictionary class that stores parameters
#more importantly it has a run method that solves and logs the task defined by its parameters


class MetaTrainer():

    #the init method needs 2 attributes:
    #d which is the dictionary of the parameters that define our problem
    #child_class which needs to be a solver class with and underlying dictionary structure
    
    def __init__(self, d, child_class):
        self.__dict__ = d

        if 'metric_res' not in d.keys():
            self.metric_res = {}

        self.child_class = child_class

    def d(self):
        non_passable_attr = ['child','child_class','metric_level','metric_res','running_log']
        return {k: v for k, v in self.__dict__.items() if k not in non_passable_attr}
    
    def d_to_child(self):
        non_passable_attr = ['child','child_class','metric_level','metric_res','running_log']
        return {k: v for k, v in self.__dict__.items() if k not in non_passable_attr}

    #run is the solver method that needs to be called
    #it formalizes the act of delegating a problem to several lower level solvers
    #it calls the run (solver) method of child_class several times 
    #while appropriatelly handling and logging the results

    def run(self):        
        self.start_signal()

        d = self.d_to_child()

        #the main part of the solving process is the following
        #we have an iterator that yields lower level attributes,
        #each of which defines a new lower level task
        for a in self.child_attr_iter():

            d.update(a)

            self.child = self.child_class(d)

            #we solve the defined lower level problem
            self.child.run()

            d.update(self.child.__dict__)

            if self.running_log:

                self.__dict__.update(self.child.d())
                self.log()

        if not self.running_log:
            self.__dict__.update(self.child.d())
            self.log()

        self.end_signal()

    def log(self):
        #self.metric_classes is shared between levels
        #self.metric_level is level-specific
        for metric_name, metric_dict in self.metric_level.items():
            metric_func = self.metric_classes[metric_dict['metric_func']](
                dict(**metric_dict['metric_params'], **self.__dict__),
                self.neptune_experiment)

            child_metric_res = self.child.metric_res[metric_dict['child_metric']]
            metric_res = metric_func(child_metric_res)
            
            #important: the way of accumulating the res is a simple +=
            #(which can mean anything depending on how we implement the metric)

            overwrite_last = metric_dict['metric_params'].get('overwrite_last',False)
            if not (hasattr(self, 'highest_level') and self.highest_level):
                if metric_name in self.metric_res.keys() and not overwrite_last:
                    try:
                        self.metric_res[metric_name] += metric_res
                    except:
                        print("ERROR saving metric:",metric_name)
                else:
                    self.metric_res[metric_name] = metric_res


    #these are the methods that need to be implemented
    #they have neutral default behaviour so skipping some of them is an option
    def start_signal(self):
        pass

    def end_signal(self):
        pass

    def child_attr_iter(self):
        return iter(())


#------------------------------------------------------------------------------
#                             SESSION LEVEL
#------------------------------------------------------------------------------
class SessionTrainer(MetaTrainer):

    highest_level = True

    def __init__(self, orig_cfg, config_fpath):
        super().__init__(d = {}, child_class = ExperimentTrainer)

        torch.manual_seed(0)

        self.serialized_cfg = cfg_parser.parse(orig_cfg)
        self.parallel = self.serialized_cfg[0]['train_params'].get('parallel', False)
        self.running_log = self.serialized_cfg[0]['train_params']['running_logs']['session']
        self.metric_level = self.serialized_cfg[0]['train_params']['metrics']['session']

        auto_naming = self.serialized_cfg[0]['train_params'].get('auto_save_path_naming', False)
        auto_save_folder = self.serialized_cfg[0]['train_params'].get('auto_save_folder', self.serialized_cfg[0]['train_params']['checkpoint_save_path'])
        if auto_naming:
            serialized_cfg = []
            for cfg in self.serialized_cfg:
                cfg_copy = copy.deepcopy(cfg)
                cfg_copy['train_params']['checkpoint_save_path'] = os.path.join(auto_save_folder,cfg_copy['experiment_name']+".pt")
                serialized_cfg.append(cfg_copy)

            self.serialized_cfg = serialized_cfg

        #--------------------CREATING THE SESSION LEVEL NEPTUNE OBJECT---------------
        with open('neptune_cfg.yaml') as file: 
            self.neptune_cfg = yaml.load(file, Loader = yaml.FullLoader)

        self.NEPTUNE_API_TOKEN = os.getenv('NEPTUNE_API_TOKEN',self.neptune_cfg['NEPTUNE_API_TOKEN'])

        if self.NEPTUNE_API_TOKEN is not None:
            if len(self.serialized_cfg)>1:
                name = f'Session: {self.serialized_cfg[0]["experiment_name_base"]}'
            else:
                name = f'Experiment: {self.serialized_cfg[0]["experiment_name"]}'

            self.neptune_experiment = neptune.init_run(
                project = self.neptune_cfg['project_qualified_name'],
                api_token = self.NEPTUNE_API_TOKEN,
                name = name,
                tags = list(self.serialized_cfg[0]["experiment_tags"])
            )
        else:
            self.neptune_experiment = MagicMock()

        self.neptune_experiment["parameters"] = str(self.serialized_cfg)
        self.neptune_experiment["yaml/"+config_fpath].upload(config_fpath)
        self.neptune_experiment.sync()

        #---------------------------------------------------------------------

    def d_to_child(self):
        non_passable_attr = ['child','child_class','metric_level','metric_res','running_log']
        if len(self.serialized_cfg)>1:
            non_passable_attr.append('neptune_experiment')
        return {k: v for k, v in self.__dict__.items() if k not in non_passable_attr}    

    def child_attr_iter(self):
        for exp_idx, experiment_cfg in enumerate(self.serialized_cfg):

            if len(self.serialized_cfg)>1:
                print( "+--------------+")
                print(f"| EXPERIMENT {exp_idx+1} |")
                print( "+--------------+")

            #Import data class
            print(f"Loading database {experiment_cfg['data_fname']} ...")
            data_lib = import_lib("process_generators",experiment_cfg["data_fname"])

            #we may want to train with random sequence length
            #in this case we have to pass the batch sizes to the data object
            train_batch_size = experiment_cfg['train_params']['train_batch_size']
            val_batch_size = experiment_cfg['train_params']['val_batch_size']

            data_params = experiment_cfg['data_params']
            dataset = data_lib.Database(data_params, train_freq = train_batch_size, val_freq = val_batch_size)
            database = dataset.database

            extra = data_params.get('extra_batch_size', 1)

            self.inference = data_params.get('inference', False)
            
            #Load model
            
            state_dict_path = experiment_cfg["model_checkpoint_path"]
            
            if experiment_cfg['train_params'].get('load_previous', False) and exp_idx>0:
                previous_cfg = self.serialized_cfg[exp_idx-1]
                state_dict_path = previous_cfg['train_params']["checkpoint_save_path"]

            print(f'Loading model {experiment_cfg["model_fname"]} (pretrained weights {state_dict_path})...')

            model_lib = import_lib("models",experiment_cfg["model_fname"])
            
            if experiment_cfg["model_params"] is None:
                experiment_cfg["model_params"] = {}
            model = to_cuda(model_lib.Model(experiment_cfg["model_params"], state_dict_path))
            if self.parallel:
                model = torch.nn.DataParallel(model)

            #Train model on data
            yield dict(
                **experiment_cfg['train_params'],
                inference = self.inference,
                extra_batch_size = extra,
                model = model,
                exp_idx = exp_idx,
                database = database
            )
            
        if self.NEPTUNE_API_TOKEN is not None:
            print('Stopping neptune session experiment...')
            self.neptune_experiment.stop()
            print('Neptune session experiment stoped.')
        
#------------------------------------------------------------------------------
#                             EXPERIMENT LEVEL
#------------------------------------------------------------------------------

#now it's time to subclass the MetaTrainer class to make it DL specific
#for this purpose we define 3 subclasses: ExperimentTrainer, EpochTrainer, BatchTrainer
#they have the following relation: ExperimentTrainer ---> EpochTrainer ----> BatchTrainer
#where A ---> B means that B is the child_class of A.
#that is in the run method of A the lower level tasks are solved by B

#In other words to compute a whole session we need to compute several epochs
#to compute an epoch we need to compute several batches


#let's start with the experiment level
class ExperimentTrainer(MetaTrainer):
    def __init__(self, d):
        #as mentioned earlier, the child_class is EpochTrainer
        super().__init__(d = d, child_class = EpochTrainer)

        self.running_log = self.running_logs['experiment']
        self.metric_level = self.metrics["experiment"]
        self.train_pairs = self.database[0]
        self.val_pairs = self.database[1]

        self.optimizer_class = string_import(self.optimizer_class)
        self.optimizer = self.optimizer_class(self.model.parameters(), self.lr)

        self.loss_fun = string_import(self.loss_fun_class)()

        
        metric_names=[]
        for metric_level in self.metrics.values():
            metric_names+=[metric["metric_func"] for metric in metric_level.values()]

        self.metric_classes = import_metric(metric_names)


        #------------CREATING THE EXPERIMENT LEVEL NEPTUNE OBJECT----------
        with open('neptune_cfg.yaml') as file: 
            self.neptune_cfg = yaml.load(file, Loader = yaml.FullLoader)
        self.exp_name = self.serialized_cfg[self.exp_idx]["experiment_name"]

        if len(self.serialized_cfg)>1:
            if self.NEPTUNE_API_TOKEN is not None: 
                self.neptune_experiment = neptune.init_run(
                    project = self.neptune_cfg['project_qualified_name'],
                    api_token = self.NEPTUNE_API_TOKEN,
                    name = f"Experiment {self.exp_idx+1}: {self.exp_name}",
                    tags = list(self.serialized_cfg[self.exp_idx]["experiment_tags"])
                )
            else:
                self.neptune_experiment = MagicMock()
        
        self.neptune_experiment["parameters"] = str(self.serialized_cfg[self.exp_idx])
        #------------------------------------------------------------------

    def start_signal(self):
        self.start_time = time.time()
        print('The experiment has started...')

        if not self.skip_train:
            self.progressbar=tqdm(range(self.num_epochs))

    def end_signal(self):
        time_ellapsed = time.time() - self.start_time
        print('The experiment has ended.')
        if len(self.serialized_cfg)>1 and self.NEPTUNE_API_TOKEN is not None:
            print('Stopping neptune experiment...')
            self.neptune_experiment.stop()
            print('Neptune experiment stoped.')
        
        print(pretty_time(time_ellapsed))
        if not self.skip_train:
            self.progressbar.close()

    def d(self):
        non_passable_attr = ['child','child_class','metric_level','metric_res','running_log']
        if len(self.serialized_cfg)>1:
            non_passable_attr.append('neptune_experiment')
        return {k: v for k, v in self.__dict__.items() if k not in non_passable_attr}

    #the iterated child attribute is the epoch number and the train bool
    #this corresponds to the fact the lower level solver operates on epochs
    #the iterator also yields the train bool which tells the EpochTrainer when to train
    #and when to validate (this way we can easily skip validation or train if needed)
    def child_attr_iter(self):
        if self.skip_train:
            epoch_iter = range(1, 2 * self.num_epochs, 2)

        elif self.skip_val:
            epoch_iter = range(0, 2 * self.num_epochs, 2)

        else:
            epoch_iter = range(0, 2 * self.num_epochs, 1)

        return ({'curr_epoch': e, 'train': e % 2 == 0} for e in epoch_iter)

#------------------------------------------------------------------------------
#                            EPOCH LEVEL
#------------------------------------------------------------------------------

#now the epoch level
class EpochTrainer(MetaTrainer):
    def __init__(self, d):
        super().__init__(d = d, child_class = BatchTrainer)

        self.batch_count = 0
        self.running_log = self.running_logs['epoch']
        self.metric_level = self.metrics["epoch"]

        if self.change_seed:
            np.random.seed(random.randrange(2**32))

        #set whether to calculate grads
        for p in self.model.parameters():
            p.requires_grad = self.train

        #set train/eval mode
        self.model.train(self.train)

        #freeze certain modules of the neural net if needed
        if self.train and hasattr(self, 'freeze_list'): 
            for a in self.freeze_list:
                module = getattr(self.model, a)
                for p in module.parameters():
                    p.requires_grad = False
                setattr(self.model, a, module)

    def start_signal(self):
        if self.train:
            self.progressbar.update()
        else:
            if self.inference:
                print('Inference:')
            else:
                print('Evaluation:')
            num_batches=math.ceil(self.serialized_cfg[self.exp_idx]["data_params"]["epoch_length"] / self.serialized_cfg[self.exp_idx]["train_params"]["val_batch_size"])
            self.epoch_progressbar=tqdm(range(num_batches))
            

    def end_signal(self):
        if self.checkpoint_save_path is not None:
            if self.parallel:
                torch.save(self.model.module.state_dict(), self.checkpoint_save_path)
            else:
                torch.save(self.model.state_dict(), self.checkpoint_save_path)

        if self.train:
            self.progressbar.set_postfix(loss = '{:06.8f}'.format(self.metric_res["epoch_loss"][-1]))

        if hasattr(self, 'adaptive_distr') and self.adaptive_distr:
            new_distr = CalcDistr(self.metric_res['ep_real_inferred'][-1],
                               local_radius = self.local_radius)
            self.train_pairs.change_gen(new_distr)

        if not self.train:
            self.epoch_progressbar.close()

            if not self.inference:
                print(f"Loss={self.metric_res['epoch_loss'][-1]:06.8f}")

    #for the initial update we mainly need to set the model to the correct mode
    #we pass the batches down to the next level because the lower level solver
    #BatchTrainer operates on batches

    def child_attr_iter(self):

        pairs = self.train_pairs if self.train else self.val_pairs
        batch_size = self.train_batch_size if self.train else self.val_batch_size

        return ({'batch': b} for b in data.DataLoader(
            dataset = pairs,
            batch_size = batch_size,
            num_workers = self.num_cores,
            shuffle = self.shuffle
        ))


#------------------------------------------------------------------------------
#                                BATCH LEVEL
#------------------------------------------------------------------------------
#this is the lowest, batch level 
class BatchTrainer(MetaTrainer):

    def __init__(self, d):
        
        super().__init__(d = d, child_class = None)

        self.running_log = self.running_logs['batch']

        if 'batch' in self.metrics:
            self.metric_level = self.metrics['batch']
        else:
            self.metric_level = {}

    #for now this level implements a different run method
    #but later it will be refactored and won't need a different run method

    def run(self):
        if not self.train:
            self.epoch_progressbar.update()
        self.batch_count += 1
        if len(self.batch)==2:
            batch_input, true_params = self.batch
        else:
            batch_input, true_params = self.batch, None

        if self.extra_batch_size > 1:
            extra_batch_size = self.extra_batch_size
            input_batch_size = batch_input.size(0)
            label_batch_size = true_params.size(0)
            batch_input = batch_input.view(extra_batch_size * input_batch_size, -1)
            true_params = true_params.view(extra_batch_size * label_batch_size, -1)

        if not hasattr(self.model, 'baseline') or not self.model.baseline:
            batch_input = to_cuda(batch_input)
            if not self.inference:
                true_params = to_cuda(true_params)

        inferred_params = self.model(batch_input)

        if not self.inference:
            k = true_params.size(1)
            loss = self.loss_fun(true_params[:, :k], inferred_params)
               
            if self.train:
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

            if 'batch_loss' in self.metric_res.keys():
                self.metric_res['batch_loss'] += [loss.item()]
            else:
                self.metric_res['batch_loss'] = [loss.item()]

        elif self.train:
            raise ValueError(f"Training is not possible without target labels!")

        for metric_name, metric_dict in self.metric_level.items():
            metric_func_name = metric_dict['metric_func']
            metric_params = metric_dict['metric_params']

            metric_func = self.metric_classes[metric_func_name](dict(**metric_params, **self.__dict__), self.neptune_experiment)

            metric_res = metric_func(true_params, inferred_params, batch_input)

            #important: the way of accumulating the res is a simple +=
            if metric_name in self.metric_res.keys():
                self.metric_res[metric_name] += metric_res
            else:
                self.metric_res[metric_name] = metric_res

#-----------------------------------------------------------------------------
#                             HELPER FUNCTIONS
#-----------------------------------------------------------------------------

def to_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var

def import_lib(location,fname):
    spec = importlib.util.spec_from_file_location("lib", os.path.join(location,fname))
    lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lib)
    return lib

def import_metric(metric_names):
    if not metric_names:
        return []

    mdl_list = [importlib.import_module('metrics.' + name)
                                                     for name in metric_names]

    class_names = [name.capitalize() for name in metric_names]

    metrics =  [getattr(mdl, class_name) for mdl, class_name 
                                                 in zip(mdl_list, class_names)]

    return {k: v for k, v in zip(metric_names, metrics)}

class CalcDistr():
    def __init__(self, real_inferred, local_radius = 0.0025, lower_bound = 0, 
                 upper_bound = 1, uniform_chance = 0.05):
        self.local_radius = local_radius
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.uniform_chance = uniform_chance
        self.real = real_inferred[0]
        self.inferred = real_inferred[1]
        self.losses = [(r - i) ** 2 for r, i in zip(self.real, self.inferred)]
        self.max_loss = max(self.losses)

    def __call__(self):

        if random.uniform(0, 1) < self.uniform_chance:
            gen_val = random.uniform(self.lower_bound, self.upper_bound)
        else:
            gen_accepted = False
            while not gen_accepted:
                r_idx = random.randint(0, len(self.real) - 1)
                if self.losses[r_idx] >= random.uniform(0, self.max_loss):
                    gen_accepted = True
                    gen_val = random.uniform(
                        max(0, self.real[r_idx] - self.local_radius),
                        min(1, self.real[r_idx] + self.local_radius))

        return gen_val

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

#if run as a script
if __name__ == "__main__":
    #parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configfile', required=True, help="File path for configfile .yaml")

    args = parser.parse_args()
    
    print('Pytorch version: ' + str(torch.__version__))
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        device = torch.cuda.current_device()
        print(f"Using {n} {torch.cuda.get_device_name(device)}{('s' if n > 1 else '')}")
    else:
        device="cpu"
        print('Warning: GPU device not found, using CPU!')

    orig_cfg = cfg_parser.read_file(args.configfile)
    SessionTrainer(orig_cfg, args.configfile).run()
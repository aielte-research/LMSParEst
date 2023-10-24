import torch
import importlib
import os

use_cuda = torch.cuda.is_available()
def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var

def string_import(s):
    parts = s.split('.')

    module = '.'.join(parts[:-1])
    attr = parts[-1]

    module = importlib.import_module(module)

    return getattr(module, attr)

#def import_model(model_class, model_name):
#
#    mdl = importlib.import_module('models.' + model_class + '.' +  model_name)
#
#    return mdl.model

def import_model(model_fname, model_params, model_state_dict_path=None):
    spec = importlib.util.spec_from_file_location("model_lib", os.path.join("models",model_fname))
    model_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_lib)

    model = model_lib.Model(model_params, model_state_dict_path)
    return to_cuda(model)

def get_state_dict(state_dict):
    if torch.cuda.is_available():
        return torch.load(state_dict)
    return torch.load(state_dict,map_location=torch.device('cpu'))

def get_state_dict_parallel(state_dict): #if it was accidentally saved as a parallel model 
    from collections import OrderedDict
    state_dict=get_state_dict(state_dict)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] #remove 'module'
        new_state_dict[name] = v
    return new_state_dict

# def save_model(model, private_save_path):
#     if private_save_path is not None:
#         torch.save(model.module.state_dict(), private_save_path)

def import_database(name, config):
    mdl = importlib.import_module('databases.' + name)
    db = mdl.Database(config)
    return db.database

def import_metric(metric_names):
    if not metric_names:
        return []

    mdl_list = [importlib.import_module('metrics.' + name)
                                                     for name in metric_names]

    class_names = [name.capitalize() for name in metric_names]

    metrics =  [getattr(mdl, class_name) for mdl, class_name 
                                                 in zip(mdl_list, class_names)]

    return {k: v for k, v in zip(metric_names, metrics)}

def import_logger(logger_names):
    if not logger_names:
        return []

    mdl_list = [importlib.import_module('loggers.' + name)
                                                     for name in logger_names]

    class_names = [name.capitalize() for name in logger_names]

    loggers =  [getattr(mdl, class_name) for mdl, class_name 
                                                 in zip(class_names, mdl_list)]

    return {k: v for k, v in zip(metric_names, metrics)}


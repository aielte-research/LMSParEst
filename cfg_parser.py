import os
import json
import yaml
import itertools
from typing import Union

# Recursively makes a list of experiments from a dictionary of config settings.
# If it finds a list somewhere in the dictionary it expands it:
# For example:
#   {'a': [1,2], 'b': 3} 
# Results in:
#   [
#     {'a': 1, 'b': 3},
#     {'a': 2, 'b': 3}
#   ]
#
# And:
#   {'a': [1,2], 'b': [3,4]}
# Results in:
#   [
#     {'a': 1, 'b': 3},
#     {'a': 1, 'b': 4},
#     {'a': 2, 'b': 3},
#     {'a': 2, 'b': 4}
#   ]
# If lists are needed inside the config, but no expansion, use tuples instead

def is_float(a_string):
    try:
        float(a_string)
        return True
    except ValueError:
        return False

def dict_parser(x):
    if type(x) is dict:
        children = [dict_parser(val) for val in x.values()]
        return [dict(zip(x.keys(), tup)) for tup in itertools.product(*children)]
    elif type(x) is list:
        return list(itertools.chain(*map(dict_parser, x)))
    else:
        return [x]

def dict_access_multi(dic, keys):
    if len(keys) == 0:
        return dic
    return dict_access_multi(dic[keys[0]], keys[1:])

def read_file(fpath):
    with open(fpath) as f:
        extension = os.path.splitext(fpath)[1].strip(".")
        if extension == 'json':
            orig = json.loads(f.read())
        elif extension in ['yaml', 'yml']:
            orig = yaml.load(f, Loader = yaml.FullLoader)
        else:
            raise ValueError(f"Config extension unrecognized: '.{extension}'. Must be '.yaml' or '.json'!")
    return orig

def parse(config_data:Union[list,dict]=None, fpath: str=None): 
    if fpath is not None:
        orig = read_file(fpath)
    elif config_data is not None:
        orig = config_data
    else:
        raise ValueError("Config must be given either as a file or as a dictionary!")

    serialized=dict_parser(orig)
    for cfg in serialized:
        if "experiment_name" in cfg:
            cfg["experiment_name"]+=cfg["experiment_name_base"]
            for ext in cfg["experiment_name_extension"]:
                val=str(dict_access_multi(cfg, ext.split("/")))
                if not is_float(val):
                    val=val.split("/")[-1].split(".")[0]
                cfg["experiment_name"]+=f'_{ext.split("/")[-1]}-{val}'   
            
    return serialized        
import json
import yaml
import itertools

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

def parse(fname): 
    with open(fname) as f:
        extension = fname.split('.')[-1]
        if extension == 'json':
            orig = json.loads(f.read())
        elif extension in ['yaml', 'yml']:
            orig = yaml.load(f, Loader = yaml.FullLoader)
        else:
            print("Config extension unknown:", extension)
            assert(False)

    serialized=dict_parser(orig)
    for cfg in serialized:
        cfg["experiment_name"]+=cfg["experiment_name_base"]
        for ext in cfg["experiment_name_extension"]:
            val=str(dict_access_multi(cfg, ext.split("/")))
            if not is_float(val):
                val=val.split("/")[-1].split(".")[0]
            cfg["experiment_name"]+=f'_{ext.split("/")[-1]}-{val}'   
            
    return serialized, orig
        
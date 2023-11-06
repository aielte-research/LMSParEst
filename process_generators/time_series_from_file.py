import os
import torch
import pandas as pd
import json

class TSPairs(torch.utils.data.dataset.Dataset):  
    def __init__(self, params):
        self.target_param_idx = params.get('target_param_idx', None)
        self.seq_start_idx = params.get('seq_start_idx', 0)        

        fpath = params.get('fpath')
        extension = os.path.splitext(fpath)[1].strip(".")
        if extension == 'csv':
            df = pd.read_csv(params.get('fpath'), header=None, index_col=False)
            data = df.to_numpy()
        elif extension == 'tsv':        
            df = pd.read_csv(params.get('fpath'), header=None, index_col=False, delim_whitespace=True)
            data = df.to_numpy()
        elif extension == 'json':
            with open(fpath) as f:
                data = json.loads(f.read())
        else:
            raise ValueError(f"Config extension unrecognized: '.{extension}'. Must be '.csv', '.tsv' or '.json'!")
        
        params["epoch_length"]=len(data)
        if params.get('n') is None:
            params["n"] = min(len(seq)-self.seq_start_idx for seq in data)
        self.seq_end_idx = self.seq_start_idx + params.get('n')
        
        self.data_tensor = torch.FloatTensor(data)
                      
    def __getitem__(self, idx):
        #this method is needed by the dataloader
        #it will be called several million times during the training session
        #each call will return a random series and an appropriate label

        #get random values from the generators
        label = self.data_tensor[idx, self.target_param_idx].unsqueeze(0)
        series = self.data_tensor[idx, self.seq_start_idx:self.seq_end_idx]

        return series, label

    def __len__(self):
        return self.data_tensor.size(0)

class Database():
    def __init__(self, config, train_freq, val_freq):
        pairs = TSPairs(config)
        self.database = (pairs, pairs)
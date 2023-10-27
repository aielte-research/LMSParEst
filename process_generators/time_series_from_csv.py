import torch
import numpy as np
import pandas as pd


#df = pd.read_csv('process_generators/fbm.csv', delim_whitespace=True)
#df = df.to_numpy()
#df = torch.FloatTensor(df)
#print(df.size())

class TSPairs(torch.utils.data.dataset.Dataset):
  
    def __init__(self, config):

        self.seq_start_idx = config['seq_start_idx']
        self.target_param_idx = config['target_param_idx']
        self.seq_end_idx = self.seq_start_idx + config['n']
        path = 'process_generators/' + config['file_name']
        df = pd.read_csv(path, delim_whitespace=True)
        df = df.to_numpy()
        self.data_tensor = torch.FloatTensor(df)

                      
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

        train_pairs = TSPairs(config)
        self.database = (train_pairs, train_pairs)


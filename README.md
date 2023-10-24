<!-- GETTING STARTED -->

### Prerequisites

* OPTIONAL: Neptune.ai account

### Installation

1. Install the conda environment (inside the folder containing `environment.yml`)
   ```sh
   conda env create
   ```
2. Activate the environment
   ```sh
   conda activate PEwDL
   ```
3. OPTIONAL: Set up neptune.ai api token and project name
   
   Put the neptune.ai project name and your api token in the `neptune_cfg.yaml` file.
   For the api token you can also use the `NEPTUNE_API_TOKEN` env. var.
   ```sh
   export NEPTUNE_API_TOKEN="YOUR_API_TOKEN"
   ```

   You can also use the technical user already provided in `neptune_cfg.yaml`, in this case no steps are needed.
   
<!-- USAGE EXAMPLES -->
## Training

```sh
   CUDA_VISIBLE_DEVICES=0 python run.py -c configs/<config_name>.yaml
```

E.g.

```sh
   CUDA_VISIBLE_DEVICES=0 python run.py -c configs/FBM/save_models_n/800.yaml
```

You can specify the save destination in the `train_params/private_save_path` field of the config file.

## Evaluating

You can load a previously trained model using the `model_state_dict_path` field of the config.
The model parameters have to be matching.

You can also evaluate the trained model on your own sequences.

E.g.

```sh
   CUDA_VISIBLE_DEVICES=0 python run.py -c configs/FBM/fBm_Hurst_LSTM_eval_from_csv.yaml
```

See the `data_params` field of the `fBm_Hurst_LSTM_eval_from_csv.yaml` config filr for more information on input csv format.

<!-- KNOWN ISSUES -->
## Known Issues
This issue was happening in our testing environment.
If getting the following error:
```sh
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found ...
```
The following might fix it for Miniconda:
```sh
export LD_LIBRARY_PATH=/home/<USERNAME>/miniconda3/lib
```
And for Anaconda:
```sh
export LD_LIBRARY_PATH=/home/<USERNAME>/anaconda3/envs/PEwDL/lib
```

If so, you can put this in your `.bashrc` file.

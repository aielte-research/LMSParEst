# Parameter Estimation of Long Memory Stochastic Processes with Deep Neural Networks

> This paper is currently under double-blind review.

## Table of Contents

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
    - [Training](#training)
    - [Inference](#inference)
- [Datasets](#datasets)
- [Models](#models)
    - [1D CNN](#1d-cnn)
    - [LSTM](#lstm)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [References](#references)
---

## Introduction

We introduce a pure deep neural network-based methodology for the estimation of long memory parameters associated with time series models, emphasizing on long-range dependence. Such parameters, notably the Hurst exponent, play an essential role in defining the long-range dependence, roughness, and self-similarity of stochastic processes. Our approach is pivotal in diverse domains such as finance, physics, and engineering, where rapid and precise estimation of these parameters is crucial.

## Getting Started

### Prerequisites

* OPTIONAL: Neptune.ai account

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/aielte-research/LMSParEst.git
   ```

2. Navigate to the project directory:
   ```sh
   cd LMSParEst
   ```

3. Install the required packages:

   * **Option a: using conda**

      First, create the conda environment:
      ```sh
      conda env create
      ```
      Activate the environment:
      ```sh
      conda activate LMSParEst
      ```
   * **Option b: using pip**
      ```sh
      pip install -r requirements.txt
      ```

4. OPTIONAL: Neptune.ai

   * **Option a: Set up neptune.ai api token and project name**
   
      Put the neptune.ai project name and your api token in the `neptune_cfg.yaml` file.
      For the api token you can also use the `NEPTUNE_API_TOKEN` env. var.
      ```sh
      export NEPTUNE_API_TOKEN="YOUR_API_TOKEN"
      ```

   * **Option b: do nothing**
      Use the technical user already provided in `neptune_cfg.yaml`, in this case no steps are needed.

   * **Option c: bypass logging to neptune.ai entirely**
   
      Edit `neptune_cfg.yaml`, write:
      ```yaml
      NEPTUNE_API_TOKEN: null
      ```
      this will disable neptune.ai logging.
   
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

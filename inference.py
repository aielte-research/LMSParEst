import argparse
import cfg_parser
from main_trainer import SessionTrainer

#if run as a script
if __name__ == "__main__":
    #parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--inputfile', required=True, default=None, type=str, help="File path of the input '.csv', '.tsv' or '.json' file.")
    required.add_argument('-o', '--outputfile', required=True, default=None, type=str, help="File path of the '.csv', '.tsv' or '.json' file the outputs will be saved in.")
    parser.add_argument('-t', '--targetcolumn', required=False, default=False, action='store_true', help="If the flag is present, column 0 of the input file is expected to contain the already known values of the target parameter.")
    parser.add_argument('-s', '--seriestype', required=False, default="fBm", type=str, help="Type of the predicted sequence. Options: 'fBm' (Hurst), 'fOU' (Hurst) and 'ARFIMA' (d).")
    parser.add_argument('-m', '--modeltype', required=False, default="LSTM", type=str, help="Type of the prediction model. Options: 'LSTM' and 'conv1D'.")
    parser.add_argument('-w', '--weightsfile', required=False, default=None, type=str, help="File path of the trained model weights. If desired to change the default which comes from the model and sequence selection.")
    parser.add_argument('-b', '--batchsize', required=False, default=32, type=int, help="Inference batch size.")
    parser.add_argument('-c', '--configfile', required=False, default=None, type=str, help="File path of the configfile '.yaml'. If desired to overwrite the default which comes from the model and sequence selection.")
    args = parser.parse_args()

    if args.configfile is None:
        seriestype = args.seriestype.upper().replace("FBM", "fBm").replace("FOU", "fOU")
        modeltype = args.modeltype.upper().replace("CONV1D", "conv1D")

        if seriestype in ["fBm", "fOU"]:
            param_name = "Hurst"
        elif seriestype=="ARFIMA":
            param_name = "d"
        else:
            raise ValueError(f"Series type '{args.seriestype}' not recognized! Available options: 'fBm', 'fOU' and 'ARFIMA'")        
        
        if not modeltype in ["LSTM", "conv1D"]:
            raise ValueError(f"Model type '{args.modeltype}' not recognized! Available options: 'LSTM' and 'conv1D'")
        
        if (seriestype=="fOU" or seriestype=="ARFIMA") and modeltype=="conv1D":
            raise ValueError(f"Option '{modeltype}' not implemented for '{seriestype}' yet!")

        config_fpath=f"configs/eval/{seriestype}/{seriestype}_{param_name}_{modeltype}_eval_from_file.yaml"       
    else:
        config_fpath=args.configfile

    orig_cfg = cfg_parser.read_file(config_fpath)

    def set_params_in_config(cfg, args):
        cfg["data_params"]["fpath"]=args.inputfile
        cfg["train_params"]["metrics"]["session"]["export_results"]["metric_params"]["fpath"]=args.outputfile
        cfg["train_params"]["train_batch_size"]=args.batchsize
        cfg["train_params"]["val_batch_size"]=args.batchsize
        if args.targetcolumn:
            cfg["data_params"]["target_param_idx"]=0
            cfg["data_params"]["seq_start_idx"]=1
            cfg["data_params"]["inference"]=False
        else:
            cfg["data_params"]["target_param_idx"]=None
            cfg["data_params"]["seq_start_idx"]=0
            cfg["data_params"]["inference"]=True
            cfg["train_params"]["metrics"]["session"]["export_results"]["metric_params"]["export_targets"]=False
            cfg["train_params"]["metrics"]["session"]={
                "export_results": cfg["train_params"]["metrics"]["session"]["export_results"]
            }
            cfg["train_params"]["metrics"]["experiment"]={
                "exp_real_inferred": cfg["train_params"]["metrics"]["experiment"]["exp_real_inferred"]
            }
            cfg["train_params"]["metrics"]["epoch"]={
                "ep_real_inferred": cfg["train_params"]["metrics"]["epoch"]["ep_real_inferred"]
            }
            cfg["train_params"]["metrics"]["batch"]={
                "batch_real_inferred": cfg["train_params"]["metrics"]["batch"]["batch_real_inferred"]
            }
        if args.weightsfile is not None:
            cfg["model_checkpoint_path"]=args.weightsfile
        return cfg

    if type(orig_cfg) is dict:
        orig_cfg=set_params_in_config(orig_cfg, args)
    elif type(orig_cfg) is list:
        for experiment in orig_cfg:
            experiment=set_params_in_config(experiment, args)
    else:
        raise ValueError(f"Unrecognized config format!")

    SessionTrainer(orig_cfg, config_fpath).run()
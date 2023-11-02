import argparse
import cfg_parser
from main_trainer import SessionTrainer

#if run as a script
if __name__ == "__main__":
    #parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', '--inputfile', required=True, default=None, type=str, help="File path for the input '.csv', '.tsv' or '.json' file.")
    required.add_argument('-o', '--outputfile', required=True, default=None, type=str, help="File path for the output '.csv', '.tsv' or '.json' file.")
    parser.add_argument('-s', '--seriestype', required=False, default="fBm", type=str, help="Type of the predicted time series. Options: 'fBm' (Hurst), 'fOU' (Hurst) and 'ARFIMA' (d).")
    parser.add_argument('-m', '--modeltype', required=False, default="lstm", type=str, help="Type of the prediction model. Options: 'LSTM' and 'conv1D'.")
    parser.add_argument('-w', '--weightsfile', required=False, default=None, type=str, help="File path for the trained model weights. If desired to change the default which comes from the model and sequence selection.")
    parser.add_argument('-b', '--batchsize', required=False, default=32, type=int, help="Inference batch size.")
    parser.add_argument('-c', '--configfile', required=False, default=None, type=str, help="File path for configfile '.yaml'. Only if desired to overwrite the default which comes from the model and sequence selection.")
    args = parser.parse_args()

    if args.configfile is None:
        seriestype = args.seriestype.lower()
        modeltype = args.modeltype.lower()
        if seriestype=="fbm":
            if modeltype=="lstm":
                config_fpath="configs/eval/fBm/fBm_Hurst_LSTM_eval_from_file.yaml"
            elif modeltype=="conv1d":
                raise ValueError(f"Option '{modeltype}' not implemented for '{seriestype}' yet!")
            else:
                raise ValueError(f"Model type '{args.modeltype}' not recognized! Available options: 'LSTM' and 'conv1D'")  
        elif seriestype=="fou":
            if modeltype=="lstm":
                raise ValueError(f"Option '{modeltype}' not implemented for '{seriestype}' yet!")
            elif modeltype=="conv1d":
                raise ValueError(f"Option '{modeltype}' not implemented for '{seriestype}' yet!")
            else:
                raise ValueError(f"Model type '{args.modeltype}' not recognized! Available options: 'LSTM' and 'conv1D'")
        elif seriestype=="arfima":
            if modeltype=="lstm":
                raise ValueError(f"Option '{modeltype}' not implemented for '{seriestype}' yet!")
            elif modeltype=="conv1d":
                raise ValueError(f"Option '{modeltype}' not implemented for '{seriestype}' yet!")
            else:
                raise ValueError(f"Model type '{args.modeltype}' not recognized! Available options: 'LSTM' and 'conv1D'")
        else:
            raise ValueError(f"Series type '{args.seriestype}' not recognized! Available options: 'fBm', 'fOU' and 'ARFIMA'")        
    else:
        config_fpath=args.configfile

    orig_cfg = cfg_parser.read_file(config_fpath)
    if type(orig_cfg) is dict:
        orig_cfg["data_params"]["fpath"]=args.inputfile
        orig_cfg["train_params"]["metrics"]["session"]["export_results"]["metric_params"]["fpath"]=args.outputfile
        orig_cfg["data_params"]["train_batch_size"]=args.batchsize
        orig_cfg["data_params"]["val_batch_size"]=args.batchsize
        if args.weightsfile is not None:
            orig_cfg["model_checkpoint_path"]=args.weightsfile
    elif type(orig_cfg) is list:
        for experiment in orig_cfg:
            experiment["data_params"]["fpath"]=args.inputfile
            experiment["train_params"]["metrics"]["session"]["export_results"]["metric_params"]["fpath"]=args.outputfile
            experiment["data_params"]["train_batch_size"]=args.batchsize
            experiment["data_params"]["val_batch_size"]=args.batchsize
            if args.weightsfile is not None:
                experiment["model_checkpoint_path"]=args.weightsfile
    else:
        raise ValueError(f"Unrecognized config format!")

    SessionTrainer(orig_cfg, config_fpath).run()
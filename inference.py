import argparse
import cfg_parser
from main_trainer import SessionTrainer

#if run as a script
if __name__ == "__main__":
    #parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configfile', required=False, default=None, type=str, help="File path for configfile '.yaml'")
    parser.add_argument('-f', '--inputfile', required=True, default=None, type=str, help="File path for the input '.csv', '.tsv' or '.json' file")
    args = parser.parse_args()

    if args.configfile is None:
        config_fpath="configs/eval/fBm/fBm_Hurst_LSTM_eval_from_file.yaml"
    else:
        config_fpath=args.configfile

    orig_cfg = cfg_parser.read_file(config_fpath)
    if type(orig_cfg) is dict:
        orig_cfg["data_params"]["fpath"]=args.inputfile 
    elif type(orig_cfg) is list:
        for exp in orig_cfg:
            exp["data_params"]["fpath"]=args.inputfile
    else:
        raise ValueError(f"Unrecognized config format!")

    SessionTrainer(orig_cfg, config_fpath).run()
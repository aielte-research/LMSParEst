import sys,argparse
import main_trainer

# parse command line arguments
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--configfile', required=True, help="configfile")

    args = parser.parse_args()
    
    main_trainer.SessionTrainer(args.configfile).run()
    
if __name__ == "__main__":
    main(sys.argv[1:])

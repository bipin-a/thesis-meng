from datetime import datetime
import argparse
import yaml
from pprint import pprint
from model_training import model_training_pipeline

def main(configs, args):
    current_time = datetime.now()
    model_tuning_configs = configs.get('base_models')
    model_training_pipeline(model_tuning_configs, args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cuda','cpu'], required=True)
    parser.add_argument('--config-file', type=str, help='Config file for the run.', required=True)
    args = parser.parse_args()

    if args.device=="cuda":
        try:
            torch.cuda.is_available() == True
        except Exception as e:
            print(e)
            raise


    with open(args.config_file, "r") as b:
        try:
            experiments = yaml.safe_load_all(b)
        except yaml.YAMLError as exc:
            print(exc)
            raise
        
        for experiment in experiments:    
            for experiment_name, config in experiment.items():
                print(experiment_name)
                main(config, args)

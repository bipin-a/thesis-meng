from pydantic.utils import deep_update
import torch
from datetime import datetime
import argparse
import yaml
from pprint import pprint
from model_training import model_training_pipeline
from adversarial_examples import adversarial_examples_pipeline

def main(configs, args):

        
    experiment_name = configs.get('experiment_name')
    
    MODEL_ROOT = lambda experiment_name, model_name: f"{experiment_name}/{model_name}/checkpoint/"
    ADV_DATASET_ROOT = lambda experiment_name, model_name, dataset_name: f"{experiment_name}/{model_name}/{attack_name}/{dataset_name}/"
    ADV_INFERENCE_RESULTS_ROOT = lambda experiment_name, model_name, attack_name, dataset_name: f"{experiment_name}/{model_name}/{attack_name}/{dataset_name}/results/"
 
    with open(write_config_path, 'r') as f:
        json.dumps(configs, f)
    
    language_models = model_training_pipeline(configs, MODEL_ROOT, args)
    dataset_names = adversarial_examples_pipeline(configs, ADV_DATASET_ROOT, language_models)
    inference_pipeline(config, language_models, dataset_names, ADV_DATASET_ROOT, ADV_INFERENCE_RESULTS_ROOT)
    # fidelity_pipeline()
    # transferability_pipeline()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cuda','cpu'], required=True)
    parser.add_argument('--config-file', type=str, help='Config file for the run.', required=True)
    args = parser.parse_args()

    current_time = datetime.now()
    current_time = current_time.strftime('%Y_%m_%d-%H_%M_%S')
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
                config.update({
                                'experiment_name':f'{experiment_name}_{current_time}'
                                })
                LR = config.get('base_models').get('tuning_params').get('LEARNING_RATE')
                LR = float(LR)
                update_LR = {'base_models':{'tuning_params':{'LEARNING_RATE': LR}}}
                config = deep_update(config, update_LR)
                print(experiment_name)
                main(config, args)

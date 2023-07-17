import json
from pydantic.utils import deep_update
from datetime import datetime
import torch
import argparse
import os
import yaml

from adversarial_examples import DatasetPipeline
from model_training import ModelPipeline
from adversarial_examples import AdversarialAttackPipeline

        # self.ADV_INFERENCE_RESULTS_ROOT = f"{self.ADV_DATASET_ROOT}/results/"

class MLExperiment:
    def __init__(self, config, device):
        self.experiment_name = config.get('experiment_name')
        self.attack_names = config.get('adv_attacks').get('names')
        self.model_config = config.get('base_models')
        self.model_names = self.model_config.get('names')
        self.dataset_config = self.model_config.get('tuning_dataset')
        self.tuning_params= self.model_config.get('tuning_params')
        self.device = device
        self.ROOT = f"{self.experiment_name}/"

        print(self.dataset_config)
        config_write_path = f'{self.ROOT}experiment.config'
        os.makedirs(os.path.dirname(config_write_path), exist_ok=True)
        
        with open(f'{self.ROOT}experiment.config', 'w') as f:
            json.dump(config, f)

    def load_dataset_pipeline(self, dataset_config):
        return DatasetPipeline(dataset_config)

    def load_model_pipeline(self, model_name):
        self.MODEL_DIR = f"{self.ROOT}{model_name}/checkpoint/"
        self.MODEL_EVAL_PATH = f"{self.MODEL_DIR}{model_name}.json"
        return ModelPipeline(model_name, self.tuning_params, self.device)

    def tune_base_models_pipeline(self, dataset_pipeline):
        # Train models
        tuned_models = []
        for model_name in self.model_names:
            model_pipeline = self.load_model_pipeline(model_name)
            model = model_pipeline()
            train_dataloader, eval_dataloader = dataset_pipeline.tokenize_load_dataset(model_name)
            tuned_model = model_pipeline.train(model, train_dataloader)
            model.save_pretrained(self.MODEL_DIR)
            results = model_pipeline.evaluate_model(model, eval_dataloader, self.MODEL_EVAL_PATH)
            print(results)
            tuned_models.append(tuned_model)
        return tuned_models

    def load_adv_attack_pipeline(self, language_model, attack_name, dataset_pipeline):
        dataset_name = dataset_pipeline.name
        raw_dataset = dataset_pipeline.raw_data
        ADV_DATASET_PATH = f"{self.ROOT}{attack_name}/{dataset_name}.csv"
        return AdversarialAttackPipeline(language_model, attack_name, raw_dataset, ADV_DATASET_PATH)

    def generate_adversarial_examples(self, attack_names, tuned_models, raw_dataset):
        adv_dataset_names = []
        for language_model in tuned_models:
            for attack_name in attack_names:
                adv_attack_pipeline = self.load_adv_attack_pipeline(language_model, attack_name, raw_dataset)
                adv_attack_pipeline.run_attack()
                adv_dataset_names.append(adv_attack_pipeline.ADV_DATASET_PATH)
        return adv_dataset_names

    def run_experiment(self):
        dataset_pipeline = self.load_dataset_pipeline(self.dataset_config)        
        tuned_models = self.tune_base_models_pipeline(dataset_pipeline)
        self.generate_adversarial_examples(self.attack_names, tuned_models, dataset_pipeline)
        # self.run_adv_examples_inference()
        # self.get_fidelity()
        # self.get_transferability()
        
def main(experiments,args):

    device = args.device
    for experiment in experiments:    
        for experiment_name, config in experiment.items():
            config.update({
                            'experiment_name':f'{experiment_name}_{current_time}'
                            })
            LR = float(config.get('base_models').get('tuning_params').get('LEARNING_RATE'))
            update_LR = {'base_models':{'tuning_params':{'LEARNING_RATE': LR}}}
            config = deep_update(config, update_LR)
            print(config.get('experiment_name'))
            experiment = MLExperiment(config, device)
            experiment.run_experiment()


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

        main(experiments, args)

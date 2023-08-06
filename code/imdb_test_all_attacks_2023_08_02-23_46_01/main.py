from transformers import AutoModelForSequenceClassification
import json
from pydantic.utils import deep_update
from datetime import datetime
import torch
import argparse
import os
import yaml

from inference import InferenceAdvExamplesPipeline, load_to_hg_data
from metrics import get_transferability

# Inference on Adv Examples Methods
def run_adv_examples_inference(tuned_models, adv_dataset_paths):
    print("Starting Inference Pipeline")
    adv_results = {}
    for tuned_model in tuned_models:
        print(tuned_model.name_or_path)
        model_name = tuned_model.name_or_path.split("/")[-2]
        print("model_name: ",model_name)
        for read_adv_dataset_path in adv_dataset_paths:
            print(read_adv_dataset_path)
            victim_model_name = read_adv_dataset_path.split("/")[-3]
            attack_name = read_adv_dataset_path.split("/")[-2]
            ADV_INFERENCE_RESULTS_ROOT = f"results/{model_name}_on_{victim_model_name}-{attack_name}_result.json"
            inference_pipeline = InferenceAdvExamplesPipeline(
                tuned_model,
                'cuda',
                ADV_INFERENCE_RESULTS_ROOT
            )
            dataset = load_to_hg_data(read_adv_dataset_path)
            tokenized_dataset = inference_pipeline.tokenize_dataset(dataset)
            results = inference_pipeline.evaluate_model(tokenized_dataset)
            key = (tuned_model.name_or_path, read_adv_dataset_path)
            adv_results[key] = results

    print("Completed Inference Pipeline")
    return adv_results


def main():
    with open('experiment.config','r') as f:
        config = json.load(f)

    attack_names = config['adv_attacks']['names']
    model_names = config['base_models']['names']

    adv_dataset_paths = [
        f"{model_name}_/{attack_name}/imdb_None.csv"
        for model_name in model_names for attack_name in attack_names
    ]
    print(adv_dataset_paths)

    model_paths = [f"{model_name}_/{model_name}/" for model_name in model_names]

    def load_model_res(file_name):
        with open(file_name, 'r') as f:
            res = json.load(f)
        return res
    tuned_results = {
        model_name: load_model_res(f"{model_name}_/{model_name}.json")
        for model_name in model_names
    }
    print(tuned_results)

    tuned_models = [
        AutoModelForSequenceClassification.from_pretrained(
            path,
            force_download=False)  for path in model_paths
    ]
    print(model_paths)
    print(tuned_models)
    adv_results = run_adv_examples_inference(
        tuned_models,
        adv_dataset_paths
    )
    print(adv_results)
    all_transferability_results = get_transferability(
        tuned_results,
        adv_results,
        model_names,
        attack_names
    )

if __name__=="__main__":
    main()

import importlib
import os
import transformers 
import textattack
import torch
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.goal_functions import UntargetedClassification
import pandas as pd
import textattack.attack_recipes as attack_recipes
from datasets import load_dataset


def load_validation_dataset():
    hg_dataset = load_dataset(path='glue', name='cola')
    dataset = textattack.datasets.Dataset(
            [(i.get("sentence"), i.get("label")) for i in hg_dataset['validation'] ]
            )
    return dataset

def wrap_language_model(model):
    model_name = model.name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    wrapped_language_model = HuggingFaceModelWrapper(model,tokenizer)
    return wrapped_language_model

def load_language_model(model_name):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_PATH + model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    wrapped_language_model = HuggingFaceModelWrapper(model,tokenizer)
    return wrapped_language_model

def run_attack(attack_model, wrapped_language_model, model_name, dataset, ADV_DATASET_ROOT):
    dataset_name = f"{attack.__name__}_{model_name}"
    write_path = f"{ADV_DATASET_ROOT}/{dataset_name}.csv"

    attack_args = textattack.AttackArgs(
        parallel = False,
        num_examples=-1,
        csv_coloring_style = 'file',
        log_to_csv=write_path,
        # checkpoint_interval=5,
        # checkpoint_dir="checkpoints",
        disable_stdout=True
    )
        
    #Building attack
    attack = attack_model.build(wrapped_language_model)
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()
    return dataset_name 

def adversarial_examples_pipeline(configs, ADV_DATASET_ROOT, language_models):
    attack_names = configs.get('adv_attacks').get('names')
    experiment_name = configs.get('experiment_name')
    data = load_validation_dataset()

    dataset_names = []
    for language_model in language_models:
        wrapped_language_model = wrap_language_model(language_model) 
        for attack_name in attack_names:
            attack = getattr(attack_recipes, attack_name)
            dataset_name = run_attack(attack, wrapped_language_model, language_model.name_or_path, data, ADV_DATASET_ROOT)
            dataset_names.append(dataset_name)
    return dataset_names


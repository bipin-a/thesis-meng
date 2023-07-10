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

MODEL_PATH = "models/checkpoints/"

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

def run_attack(attack_model, wrapped_language_model, model_name, dataset):

    attack_args = textattack.AttackArgs(
        parallel = False,
        num_examples=-1,
        csv_coloring_style = 'file',
        log_to_csv=f"adv_examples/{attack_model.__name__}_{model_name}_file.csv",
        # checkpoint_interval=5,
        # checkpoint_dir="checkpoints",
        disable_stdout=True
    )
        
    #Building attack
    attack = attack_model.build(wrapped_language_model)
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()

def adversarial_examples_pipeline(configs,language_models):

    current_time = configs.get("current_time")
    attack_names = configs.get('adv_attacks').get('names')
    data = load_validation_dataset()
 
    for language_model in language_models:
        wrapped_language_model = wrap_language_model(language_model) 
        for attack_name in attack_names:
            attack = getattr(attack_recipes, attack_name)
            run_attack(attack, wrapped_language_model, language_model.name_or_path, data)

def main():
    model_names = [f for f in os.listdir(MODEL_PATH)]
    data = load_validation_dataset()
    attacks = [
            A2TYoo2021,
            #BERTAttackLi2020,
            ]
    for model_name in model_names:
        language_model = load_language_model(model_name)
        for attack in attacks:
            print(attack.__name__)
            run_attacks(attack, language_model, model_name, data)


if __name__ == "__main__":
    print("running main")
    main()


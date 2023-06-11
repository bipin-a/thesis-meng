import os
import transformers 
import textattack
import torch
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.goal_functions import UntargetedClassification
import pandas as pd
from textattack.attack_recipes import (
    A2TYoo2021,
    BERTAttackLi2020
)
from datasets import load_dataset

MODEL_PATH = "models/checkpoints/"

def load_text_attack_dataset():
    hg_dataset = load_dataset(path='glue', name='cola')
    dataset = textattack.datasets.Dataset(
            [(i.get("sentence"), i.get("label")) for i in hg_dataset['validation'] ]
            )
    return dataset

def load_language_models(model_name):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_PATH + model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    language_model = HuggingFaceModelWrapper(model,tokenizer)
    return language_model

def run_attacks(attack_model, language_model,model_name, dataset):

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
    attack = attack_model.build(language_model)
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()
    
def main():
    model_names = [f for f in os.listdir(MODEL_PATH)]
    data = load_text_attack_dataset()
    attacks = [
            A2TYoo2021,
            #BERTAttackLi2020,
            ]
    for model_name in model_names:
        language_model = load_language_models(model_name)
        for attack in attacks:
            print(attack.__name__)
            run_attacks(attack, language_model, model_name, data)


if __name__ == "__main__":
    print("running main")
    main()


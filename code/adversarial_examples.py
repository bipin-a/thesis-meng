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



def load_text_attack_dataset():
    hg_dataset = load_dataset(path='glue', name='cola')
    dataset = textattack.datasets.Dataset(
            [(i.get("sentence"), i.get("label")) for i in hg_dataset['validation'] ]
            )
    return dataset

def load_language_models():
    checkpoint_names = ["models/default_params",]

    model = transformers.AutoModelForSequenceClassification.from_pretrained(checkpoint_names[0])
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    language_model = HuggingFaceModelWrapper(model,tokenizer)
    print(language_model)

    return language_model,"bert-cola-tuned"

def run_attacks(language_model,model_name, dataset):

    attacks = [
            (A2TYoo2021,"A2TYoo2021"),
            (BERTAttackLi2020,"BERTAttackLi2020")
            ]
    for attack_model,attack_name in attacks:         
        attack = attack_model.build(language_model)
        
        attack_args = textattack.AttackArgs(
            parallel = True,
            num_examples=-1,
            csv_coloring_style = 'html',
            log_to_csv=f"adv_examples/{attack_name}{model_name}_html.csv",
            # checkpoint_interval=5,
            # checkpoint_dir="checkpoints",
            disable_stdout=True
        )
        
    #Building attack
    attacker = textattack.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()
    
def main():
    data = load_text_attack_dataset()
    language_models, model_name = load_language_models()
    print(language_models, model_name)
    run_attacks(language_models, model_name, data)


if __name__ == "__main__":
    main()


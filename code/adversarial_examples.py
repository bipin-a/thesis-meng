import textattack
import pandas as pd
import textattack.attack_recipes as attack_recipes
from transformers import AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset as DatasetWrapper

def wrap_language_model(model, tokenizer):
    return HuggingFaceModelWrapper(model, tokenizer)

def wrap_validation_dataset(raw_dataset):
    print(raw_dataset)
    adv_eval_dataset = [ (i.get("sentence"), i.get("label")) for i in raw_dataset['validation'] ]
    print("adv eval leng: ", len(adv_eval_dataset))
    return DatasetWrapper(adv_eval_dataset)

class AdversarialAttackPipeline:
    def __init__(self, language_model, attack_name, raw_dataset, ADV_DATASET_PATH):
        '''
        Wrapping Language Model
        Wrapping Validation Dataset
        '''

        self.adv_attack_model = getattr(attack_recipes, attack_name)
        self.language_model_name = language_model.name_or_path
        self.wrapped_validation_dataset = wrap_validation_dataset(raw_dataset)
        tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
        self.wrapped_language_model = wrap_language_model(language_model, tokenizer) 
        self.ADV_DATASET_PATH = ADV_DATASET_PATH

    def run_attack(self):
        attack_args = textattack.AttackArgs(
            parallel = False,
            num_examples=-1,
            csv_coloring_style = 'file',
            log_to_csv= self.ADV_DATASET_PATH,
            disable_stdout=True,
            enable_advance_metrics=True
        )
            
        #Building attack
        attack = self.adv_attack_model.build(self.wrapped_language_model)
        attacker = textattack.Attacker(attack, self.wrapped_validation_dataset, attack_args)
        attacker.attack_dataset()



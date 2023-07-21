import textattack
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset as DatasetWrapper
import pandas as pd
import textattack.attack_recipes as attack_recipes
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class DatasetPipeline:
    def __init__(self, config):
        path_ = config.get('path')
        name_ = config.get('name')
        self.raw_data = load_dataset(path=path_, name=name_) 
        self.name = f'{path_}_{name_}'

    def tokenize_function(self, examples):
        return self.tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True
                )
    
    def tokenize_load_dataset(self, model_name):

        self.tokenized_model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenized_model_name)
        tokenized_datasets = self.raw_data.map(self.tokenize_function, batched=True)

        tokenized_datasets = tokenized_datasets.remove_columns(["idx"])
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

        print(f"tokenized dataset: {tokenized_datasets}")
        tokenized_datasets.set_format("torch")

        train_set = tokenized_datasets["train"].shuffle(seed=42)
        eval_set = tokenized_datasets["validation"].shuffle(seed=42)
        train_dataloader = DataLoader(train_set,
                                            shuffle=True,
                                            batch_size=8)
        eval_dataloader = DataLoader(eval_set,
                                            shuffle=False,
                                            batch_size=8)
        
        return train_dataloader, eval_dataloader


def wrap_language_model(model, tokenizer):
    return HuggingFaceModelWrapper(model, tokenizer)

def wrap_validation_dataset(raw_dataset):
    return DatasetWrapper(
        [ (i.get("sentence"), i.get("label")) for i in raw_dataset['validation'] ]
        )

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
            disable_stdout=True
        )
            
        #Building attack
        attack = self.adv_attack_model.build(self.wrapped_language_model)
        attacker = textattack.Attacker(attack, self.wrapped_validation_dataset, attack_args)
        attacker.attack_dataset()


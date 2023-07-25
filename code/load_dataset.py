from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class DatasetPipeline:
    def __init__(self, config):
        path_ = config.get('path')
        name_ = config.get('name')
        print(path_)
        if path_ == 'imdb':
            print('imdb')
            self.raw_data = load_dataset(path=path_).rename_column("text", "sentence")
            self.raw_data['validation'] = self.raw_data.pop('test')

            print(self.raw_data)
        else:
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

        if 'idx' in self.raw_data.column_names.get('train'):
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



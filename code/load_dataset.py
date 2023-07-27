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
    
        self.train_len = self.raw_data['train'].num_rows
        if config.get('train_size') != 'None':
            self.raw_data['train'] = self.raw_data['train'].shuffle(seed=42).select(range(config.get('train_size')))

        self.eval_len = self.raw_data['validation'].num_rows
        if config.get('eval_size') != 'None':
            self.raw_data['validation'] = self.raw_data['validation'].shuffle(seed=42).select(range(config.get('eval_size')))



    def tokenize_function(self, examples):
        return self.tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True
                )
    
    def tokenize_load_dataset(self, model_name):
        '''
        Tokenizes using pretrained tokenizer of model_name 
        Clean columns
        Shuffles and then samples using config params
        '''
        self.tokenized_model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenized_model_name)
        tokenized_datasets = self.raw_data.map(self.tokenize_function, batched=True)

        if 'idx' in self.raw_data.column_names.get('train'):
            tokenized_datasets = tokenized_datasets.remove_columns(["idx"])

        tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

        train_set = tokenized_datasets["train"]
        eval_set = tokenized_datasets["validation"]

        train_dataloader = DataLoader(train_set,
                                            shuffle=True,
                                            batch_size=8)
        eval_dataloader = DataLoader(eval_set,
                                            shuffle=False,
                                            batch_size=8)
        
        print(f"train: {train_set}")
        print(f"eval: {eval_set}")
        return train_dataloader, eval_dataloader



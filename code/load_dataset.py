from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class DatasetPipeline:
    def __init__(self, config):
        '''
        load data
        noramlize column names
        filter to min letter count
        shuffle
        sample based on train and eval size
        '''
        path_ = config.get('path')
        name_ = config.get('name')
        limit_text_size = config.get('limit_text_size')
        self.tokenize_truncation = config.get('tokenize_truncation')
        self.name = f'{path_}_{name_}'
        print(limit_text_size)
        if path_ in ('imdb','rotten_tomatoes'):
            print('imdb')
            self.raw_data = load_dataset(path=path_).rename_column("text", "sentence")
            self.raw_data['validation'] = self.raw_data.pop('test')
        else:
            self.raw_data = load_dataset(path=path_, name=name_)
        print(self.raw_data)

        # Remove all cases where text is less than text size for speed.
        self.raw_data = self.raw_data.filter(
            lambda x: len(x["sentence"]) < limit_text_size, batched=False
        )

        self.train_len = self.raw_data['train'].num_rows
        self.eval_len = self.raw_data['validation'].num_rows

        self.raw_data['train'] = self.raw_data['train'].shuffle(seed=42)
        self.raw_data['validation'] = self.raw_data['validation'].shuffle(seed=42)

        if config.get('train_size') != 'None':
            self.raw_data['train'] = self.raw_data['train'].select(
                range(config.get('train_size'))
            )
        if config.get('eval_size') != 'None':
            self.raw_data['validation'] = self.raw_data['validation'].select(
                range(config.get('eval_size'))
            )

        print(self.raw_data)

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length = 512,
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



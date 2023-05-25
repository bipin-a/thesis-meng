import pdb
import argparse
from datasets import load_dataset
import transformers 
import torch
import pandas as pd
import evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from transformers import get_scheduler
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from tqdm.auto import tqdm


def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

def load_tokenize_dataset():
    data = load_dataset(path='glue', name='cola') 
    print(data)

    print(pd.DataFrame(data["train"]).label.value_counts())
    print(pd.DataFrame(data["validation"]).label.value_counts())
    
    tokenized_datasets = data.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["idx"])
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    print(f"tokenized dataset: {tokenized_datasets}")
    
    tokenized_datasets.set_format("torch")
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)
    
    return train_dataset, eval_dataset

def train_model(train_dataset):
   
    train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=8)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased",
                                                            num_labels=2) 
    optimizer = AdamW(model.parameters(), lr=5e-5)
 
    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model.to(args.device)

    print("######### \ndevice\n######### \n", args.device)

    progress_bar = tqdm(range(num_training_steps))
    
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
    
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    

    return model

def evaluate_model(model, eval_dataset):
    
    eval_dataloader = DataLoader(eval_dataset,
                             batch_size=8)
    
    metric = evaluate.load("accuracy")
    model.eval()
    
    for batch in eval_dataloader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
    
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    metric.compute()

    return metric
    

def main():
    train_dataset, eval_dataset = load_tokenize_dataset()
    model = train_model(train_dataset) 
    model.save_pretrained(f"models/{args.model_name}")
    metric = evaluate_model(model, eval_dataset)
    print(metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('device', choices=['cuda','cpu'])
    args = parser.parse_args()
    if args.device=="cuda":
        try:
            torch.cuda.is_available() == True
        except Exception as e:
            print(e)
        
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    print('running main')
    main()

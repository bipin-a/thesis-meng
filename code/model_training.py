from datetime import datetime
import pdb
import argparse
from datasets import load_dataset
import transformers 
import torch
import pandas as pd
import evaluate
from torch.utils.data import DataLoader
import torch
from transformers import get_scheduler
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from tqdm.auto import tqdm

LEARNING_RATE = 1
NUM_EPOCHS = 1

def tokenize_function(examples, tokenizer_):
    return tokenizer_(examples["sentence"], padding="max_length", truncation=True)

def load_tokenize_dataset(tokenizer):
    data = load_dataset(path='glue', name='cola') 
    print(data)

    print(pd.DataFrame(data["train"]).label.value_counts())
    print(pd.DataFrame(data["validation"]).label.value_counts())
    tokenized_datasets = data.map(tokenize_function, batched=True, fn_kwargs={"tokenizer_": tokenizer})
    tokenized_datasets = tokenized_datasets.remove_columns(["idx"])
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    print(f"tokenized dataset: {tokenized_datasets}")
    tokenized_datasets.set_format("torch")
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200))
    eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(200))
    
    return train_dataset, eval_dataset


def model_training_loop(model, num_training_steps, train_dataloader, optimizer, lr_scheduler ):
    #model = torch.nn.DataParallel(model)
    model.to(args.device)
    
    print(f"######### \ndevice\n#########\n {args.device}")
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(NUM_EPOCHS):
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

def train_model(train_dataset,model_name):
    train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=8)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) 
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE) 
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    # TODO: Update from default
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    model = model_training_loop(model, num_training_steps, train_dataloader, optimizer, lr_scheduler)
    return model

def evaluate_model(model, eval_dataset):
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    results= metric.compute()
    return results
    

def main():
    # TODO: Assumed uncased for Glue
    current_time = datetime.now()
    model_names = ["bert-base-uncased","albert-base-v2","distilbert-base-uncased"]
    for model_name in model_names: 
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset, eval_dataset = load_tokenize_dataset(tokenizer)
        model = train_model(train_dataset, model_name) 
        model.save_pretrained(f"models/{model_name}")
        results = evaluate_model(model, eval_dataset)
        print(results)

        hyperparams = {"model": model_name, "learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS}
        evaluate.save(f"./results/{model_name}-{current_time.strftime('%Y_%m_%d-%H_%M_%S')}.json", **results, **hyperparams)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('device', choices=['cuda','cpu'])
    args = parser.parse_args()
    if args.device=="cuda":
        try:
            torch.cuda.is_available() == True
        except Exception as e:
            print(e)
    print('running main')
    main()

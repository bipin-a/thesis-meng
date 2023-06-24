from datetime import datetime
import logging
import argparse
import evaluate
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer
import pandas as pd
import os
from datasets import Dataset
from transformers import AutoModelForSequenceClassification

MODEL_PATH = "models/checkpoints/"
ADV_DATASET_PATH = "adv_examples/"


def clean_perturbed_dataset(text):
  text = text.replace('[[','')
  text = text.replace(']]','')
  return text

def tokenize_function(examples, tokenizer_):
    return tokenizer_(examples["sentence"], padding="max_length", truncation=True)

def load_adv_dataset(dataset_name):
    logging.info(f"Loading {dataset_name}")
    adv_df = pd.read_csv(f"{ADV_DATASET_PATH}{dataset_name}")[['perturbed_text','ground_truth_output']]
    adv_df = adv_df.rename({"perturbed_text":"sentence", "ground_truth_output":"labels"}, axis=1)
    adv_df['sentence'] = adv_df['sentence'].apply(clean_perturbed_dataset)
    dataset = Dataset.from_pandas(adv_df)
    return dataset

def tokenize_dataset(dataset, model_name):
    logging.info(f"Applying tokenizer for model:{model_name} dataset: {dataset}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        fn_kwargs={"tokenizer_": tokenizer}
        )
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence"])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

def evaluate_model(model, eval_adv_dataset):
    logging.info(f"Evaluating Model with model: {model} and dataset: {eval_adv_dataset}")
    eval_dataloader = DataLoader(eval_adv_dataset, batch_size=8)
    evaluation_metrics = evaluate.combine(["accuracy", "recall", "precision", "f1"])
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        evaluation_metrics.add_batch(predictions=predictions, references=batch["labels"])
    results = evaluation_metrics.compute()
    return results

def main():
    model_names = [f for f in os.listdir(MODEL_PATH)]
    dataset_names = [f for f in os.listdir(ADV_DATASET_PATH)]
    for model_name in model_names:
        for dataset_name in dataset_names:
            dataset = load_adv_dataset(dataset_name)
            eval_adv_dataset = tokenize_dataset(dataset, model_name)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH+model_name) 
            print(eval_adv_dataset)
            print(model._get_name())
            model.to(args.device)
            results = evaluate_model(model, eval_adv_dataset)
            hyperparams = {"model": model_name, "dataset_name":dataset_name}
            evaluate.save(f"models/results/adversarial_inference/{dataset_name.split('.')[0]}_{model_name}-{current_time.strftime('%Y_%m_%d-%H_%M_%S')}.json", **results, **hyperparams)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('device', choices=['cuda','cpu'])
    args = parser.parse_args()
    current_time = datetime.now()

    if args.device=="cuda":
        try:
            torch.cuda.is_available() == True
        except Exception as e:
            print(e)
    main()



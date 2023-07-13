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

def clean_perturbed_dataset(text):
  text = text.replace('[[','')
  text = text.replace(']]','')
  return text

def tokenize_function(examples, tokenizer_):
    return tokenizer_(examples["sentence"], padding="max_length", truncation=True)

def load_adv_dataset(ADV_DATASET_ROOT, dataset_name):
    logging.info(f"Loading {dataset_path}")
    adv_df = pd.read_csv(f"{ADV_DATASET_ROOT}/{dataset_name}")[['perturbed_text','ground_truth_output']]
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

def evaluate_model(model, eval_adv_dataset, args):
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

def inference_pipeline(config, language_models, dataset_names, ADV_DATASET_ROOT, ADV_INFERENCE_RESULTS_ROOT):
    experiment_name = configs.get('experiment_name')

    for language_model in language_models:
        wrapped_langauge_model = wrap_language_model(language_model)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for dataset_name in dataset_names:
            dataset = load_adv_dataset(dataset_name)
            eval_adv_dataset = tokenize_dataset(dataset, language_model.name_or_path)
            results = evaluate_model(model, eval_dataset, args)
            print(results)

            hyperparams = {"model": model_name}
            evaluate.save(f"{ADV_INFERENCE_RESULTS_ROOT}.json", **results, **hyperparams)
    print("Completed Inference Pipeline", tuned_models)

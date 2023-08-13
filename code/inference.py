import os
import numpy as np
import json
import evaluate
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset

def clean_perturbed_dataset(text):
    text = text.replace('[[','')
    text = text.replace(']]','')
    return text

def load_to_hg_data(read_adv_dataset_path):
    adv_df = pd.read_csv(read_adv_dataset_path)[['perturbed_text','ground_truth_output']]
    adv_df = adv_df.rename({"perturbed_text":"sentence", "ground_truth_output":"labels"}, axis=1)
    adv_df['sentence'] = adv_df['sentence'].apply(clean_perturbed_dataset)
    dataset = Dataset.from_pandas(adv_df)
    return dataset

class InferenceAdvExamplesPipeline:
    def __init__(self, tuned_model, device, write_adv_inference_results_root, tuned_model_name=None, write_preds_path=None):
        self.WRITE_ADV_INFERENCE_RESULTS_ROOT = write_adv_inference_results_root
        self.device = device

        self.tuned_model = tuned_model
        if tuned_model_name:
            self.model_name = tuned_model_name
        else:
            self.model_name = tuned_model.name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.write_preds_path = write_preds_path
        self.tuned_model.to(self.device)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["sentence"], padding="max_length", truncation=True)

    def tokenize_dataset(self, dataset):
        tokenized_dataset = dataset.map(self.tokenize_function,
                                        batched=True,
                                        )
        tokenized_dataset = tokenized_dataset.remove_columns(["sentence"])
        tokenized_dataset.set_format("torch")
        return tokenized_dataset

    def evaluate_model(self, eval_adv_dataset):
        all_predictions=np.array([])
        eval_dataloader = DataLoader(eval_adv_dataset, batch_size=8)
        evaluation_metrics = evaluate.combine(["accuracy", "recall", "precision", "f1"])
        self.tuned_model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.tuned_model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions = np.append(
                all_predictions,
                predictions.data.cpu().numpy()
            )
            evaluation_metrics.add_batch(predictions=predictions, references=batch["labels"])
        results = evaluation_metrics.compute()

        if self.WRITE_ADV_INFERENCE_RESULTS_ROOT:
            hyperparams = {"model": self.model_name}
            evaluate.save(self.WRITE_ADV_INFERENCE_RESULTS_ROOT, **results, **hyperparams)

        if self.write_preds_path:
            os.makedirs(os.path.dirname(self.write_preds_path), exist_ok=True)
            np.savetxt(self.write_preds_path, all_predictions, delimiter=",")
        return results, all_predictions

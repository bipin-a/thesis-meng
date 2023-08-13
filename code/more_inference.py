from transformers import AutoModelForSequenceClassification
from datasets import Dataset
import pandas as pd
from inference import InferenceAdvExamplesPipeline

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


experiment_names = [
#    'glue_2023_08_07-14_22_27',
    'imdb_2023_08_08-23_18_24',
]
data_names = [
#    'glue_cola',
    'imdb_None',
]
model_names = ['bert-base-uncased', 'albert-base-v2', 'distilbert-base-uncased']
attack_names = ['A2TYoo2021', 'TextFoolerJin2019', 'BAEGarg2019']


results = {}
all_predictions = {}
for experiment_name, data_name in zip(experiment_names, data_names):
    adv_dataset_paths = [
        f"{experiment_name}/adv_datasets/{model_name}/{attack_name}/{data_name}.csv"
        for model_name in model_names for attack_name in attack_names
    ]
    for inference_model_name in model_names:
        for read_adv_dataset_path in adv_dataset_paths:
            dataset = load_to_hg_data(read_adv_dataset_path)
            victim_model_name = read_adv_dataset_path.split("/")[-3]
            attack_name = read_adv_dataset_path.split("/")[-2]
            ADV_INFERENCE_PREDICTIONS_PATH = f"{experiment_name}/raw_predictions/{inference_model_name}_on_{victim_model_name}_{attack_name}_preds.txt"
            tuned_model = AutoModelForSequenceClassification.from_pretrained(
                f"{experiment_name}/{inference_model_name}"
            )
            inference_pipeline = InferenceAdvExamplesPipeline(
                tuned_model,
                'cuda',
                None,
                inference_model_name,
                ADV_INFERENCE_PREDICTIONS_PATH
            )
            tokenized_dataset = inference_pipeline.tokenize_dataset(dataset)
            results, predictions = inference_pipeline.evaluate_model(tokenized_dataset)
            key = (inference_model_name, read_adv_dataset_path)
            results[key] = results
            all_predictions[key] = predictions

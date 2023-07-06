import logging
import pandas as pd
import evaluate
import os
from evaluate import load


MODEL_PATH = "models/checkpoints/"
ADV_DATASET_PATH = "adv_examples/"

def clean_perturbed_dataset(text):
  text = text.replace('[[','')
  text = text.replace(']]','')
  return text

def load_adv_dataset(dataset_name):
    logging.info(f"Loading {dataset_name}")
    adv_df = pd.read_csv(f"{ADV_DATASET_PATH}{dataset_name}")[['perturbed_text','ground_truth_output']]
    adv_df = adv_df.rename({"perturbed_text":"sentence", "ground_truth_output":"labels"}, axis=1)
    adv_df['sentence'] = adv_df['sentence'].apply(clean_perturbed_dataset)
    # dataset = Dataset.from_pandas(adv_df)
    return adv_df

def load_all_adv_datasets(ADV_DATASET_PATH):
    dataset_names = [f for f in os.listdir(ADV_DATASET_PATH)]
    dfs = []
    for dataset_name in dataset_names:
      col_name = dataset_name.split('.')[0]
      df = load_adv_dataset(dataset_name)
      df.columns = [col_name+"_"+col for col in df.columns]
      dfs.append(df)

    all_adv_dataset = pd.concat(dfs,axis=1)
    print(all_adv_dataset.info())
    print(all_adv_dataset.describe())
    return all_adv_dataset

def get_bert_score(predictions,truth):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=truth, lang="en")
    return results

def main():
    all_adv_dataset = load_all_adv_datasets(ADV_DATASET_PATH)
    source = all_adv_dataset['A2TYoo2021_albert-base-v2_file_sentence'].values
    target = all_adv_dataset['A2TYoo2021_distilbert-base-uncased_file_sentence'].values
    score = get_bert_score(source, target)
    print(score)

if __name__=="__main__":
    main()


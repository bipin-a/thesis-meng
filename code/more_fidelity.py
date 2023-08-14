import os
import tensorflow_hub as hub
import torch
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util
from evaluate import load


def clean_perturbed_dataset(text):
    text = text.replace("[[", "")
    text = text.replace("]]", "")
    return text


def get_granular_fidelity(adv_dataset_path):
    adv_df = pd.read_csv(adv_dataset_path)
    adv_df["perturbed_text_clean"] = adv_df["perturbed_text"].apply(
        clean_perturbed_dataset
    )
    adv_df["original_text_clean"] = adv_df["original_text"].apply(
        clean_perturbed_dataset
    )

    perturbed = adv_df["perturbed_text_clean"]
    original = adv_df["original_text_clean"]

    embed_perturbed = sim_model.encode(perturbed, convert_to_tensor=True)
    embed_original = sim_model.encode(original, convert_to_tensor=True)

    adv_df["sentence_similarity"] = [
        util.pytorch_cos_sim(orig, perturbed).cpu().data.numpy().flatten()[0]
        for orig, perturbed in zip(embed_perturbed, embed_original)
    ]

    adv_df["bertscore_result"] = bertscore.compute(
        predictions=perturbed,
        references=original,
        lang="en",
    ).get("precision")

    use_perturbed = USE_model(perturbed).numpy()
    use_original = USE_model(original).numpy()

    adv_df["USE_similarity"] = sim_metric(
        torch.from_numpy(use_perturbed), torch.from_numpy(use_original)
    ).numpy()

    return adv_df


if __name__ == "__main__":
    sim_metric = torch.nn.CosineSimilarity(dim=1)
    USE_model = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")

    bertscore = load("bertscore", model="distilbert-base-uncased")  # TODO CHECK

    sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    experiment_names = [
        "glue_2023_08_07-14_22_27",
        "imdb_2023_08_08-23_18_24",
    ]
    data_names = [
        "glue_cola",
        "imdb_None",
    ]
    model_names = ["bert-base-uncased", "albert-base-v2", "distilbert-base-uncased"]
    attack_names = ["A2TYoo2021", "TextFoolerJin2019", "BAEGarg2019"]

    for experiment_name, data_name in zip(experiment_names, data_names):
        fidelity_results = {}
        for model_name in model_names:
            for attack_name in attack_names:
                adv_dataset_path = f"{experiment_name}/adv_datasets/{model_name}/{attack_name}/{data_name}.csv"
                fidelity_results[(model_name, attack_name)] = get_granular_fidelity(
                    adv_dataset_path
                )

                path_name = f"{experiment_name}/granular_fidelity_results.pickle"
                os.makedirs(os.path.dirname(path_name), exist_ok=True)

                with open(path_name, "wb") as f:
                    pickle.dump(fidelity_results, f)

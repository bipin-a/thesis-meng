import pandas as pd
from inference import InferenceAdvExamplesPipeline

def get_transferability(experiment_name,
                        tuned_results,
                        adv_results,
                        inference_models,
                        attacks
                        ):
    
    adv_dataset_paths = [ 
        f'{experiment_name}/{model}/{attack}/glue_cola.csv'
        for model in inference_models
        for attack in attacks
        ]

    for adv_dataset_path in adv_dataset_paths:
        for model_name in inference_models:
            adv_result = pd.Series(adv_results.get( (model_name,adv_dataset_path) ))
            og_result = pd.Series(tuned_results.get(model_name))
            print(adv_result)
            print(og_result)
            print(adv_result/og_result)

# def get_similarity():
#     InferenceAdvExamplesPipeline.load_to_hg_data()

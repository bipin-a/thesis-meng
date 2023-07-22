from evaluate import load
import pandas as pd
from inference import clean_perturbed_dataset
from sentence_transformers import SentenceTransformer, util
import json

def get_transferability(experiment_name,
                        tuned_results,
                        adv_results,
                        inference_models,
                        attacks
                        ):
    '''
    Finding drop in evaluation metrics of all models
    from performance on the original dataset to all 
    variations of perturbed datasets. 
    '''
    
    adv_dataset_paths = [ 
        f'{experiment_name}/{model}/{attack}/glue_cola.csv'
        for model in inference_models
        for attack in attacks
        ]
    transferability_results = []


    for adv_dataset_path in adv_dataset_paths:
        for model_name in inference_models:
            adv_result = pd.Series(adv_results.get( (model_name,adv_dataset_path) ))
            og_result = pd.Series(tuned_results.get(model_name))
            transferability = (adv_result - og_result)/og_result
            adv_dataset_name = '-'.join(adv_dataset_path.split('/')[1:-1])
            transferability.name = f'{model_name}_on_{adv_dataset_name}'
            
            transferability_results.append(transferability)

    all_transferability_results = pd.concast(transferability_results)
    file_name = f"{experiment_name}/metrics/transferability_results.csv"
    all_transferability_results.to_csv(file_name)
    return all_transferability_results

def get_fidelity(experiment_name, adv_dataset_paths):
    '''
    Finding similarity of the dataset for each attack
    '''
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    fidelity_results = {}
    for adv_dataset_path in adv_dataset_paths:
        adv_df = pd.read_csv(adv_dataset_path)
        adv_df['perturbed_text_clean'] = clean_perturbed_dataset(adv_df['perturbed_text'])
        
        bertscore = load("bertscore")
        
        perturbed = adv_df['perturbed_text_clean']
        original = adv_df['original_text']

        bertscore_result = bertscore.compute(predictions=perturbed, 
                                    references=original,
                                    lang="en")
        
        embed_perturbed = model.encode(perturbed, convert_to_tensor=True)
        embed_original = model.encode(original, convert_to_tensor=True)
        
        sentence_similarity = util.pytorch_cos_sim(embed_perturbed, embed_original)

        fidelity_results[adv_dataset_path] = {"bertscore_result":bertscore_result,
                                          "sentence_similarity":sentence_similarity
                                          }
        
        adv_dataset_name = '-'.join(adv_dataset_path.split('/')[1:-1])

    file_name = f"{experiment_name}/metrics/fidelity_results.json"
    with open(file_name, 'w') as f:
        json.dump(fidelity_results, f)
    return fidelity_results



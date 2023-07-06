import os
import pandas as pd
import json 

ADV_RESULTS_PATH = "models/results/adversarial_inference/"
OG_RESULTS_PATH = "models/results/"

def loadfile(path):
  with open(path, 'r') as f:
    results = json.load(f)
  return results

def load_model_results(path):
    model_results = {f[:-25]:loadfile(path+f) for f in os.listdir(path) if os.path.isfile(path+f)}
    print(model_results.keys())
    return model_results

def main():
    adv_model_results = load_model_results(ADV_RESULTS_PATH)

if __name__=="__main__":
    main()



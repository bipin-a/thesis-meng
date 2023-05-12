from datasets import load_dataset


def load_dataset():
    data = load_dataset('glue','cola', split="train[:100]+train[-100:]") 
    return data

def main():
    data = load_dataset()
    
    train_models(data)

if __name__ == 'main':
    main():

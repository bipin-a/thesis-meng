## 2023-03-01

### Old ideas: 
- *Transferability*:
  - Measure performance of different models across the same adversarial text perturbations between models. 

### New Ideas: 

- *Robustness*:
  - Adversarial text perturbations can be tailored or _"fit"_ to the model, by opitimizing which perturbations have the highest affect on the output.
  - We can quantify which models require the most perturbations of the input in order for a misclassification depending on the targetted attack.

### Action Items: 

1. Create a presentation outlining methodology and experimentations to measure robustness of a model as defined as: 

```
Avgerage or minimum measure of perturbations of the input required for a misclassification for a type of adversarial text attack. 
```

2. Start getting hands dirty with [text attack](https://textattack.readthedocs.io/en/latest/) types. 

### Open Ended Questions: 

- Why don't adverserial attacks for text transfer between models, whereas adverserial attacks for images are transferable?


## 2023-04-21

### Final Idea:
- Rank the top N adversarial attacks on language models on the basis of `Transferability` and `Semantic Similarity`. Using [Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models, Boxin Wang 2021](https://openreview.net/forum?id=GF9cSKI3A_q) and [On the Transferability of Adversarial Attacks against Neural Text Classifier, Liping Yuan 2021](https://aclanthology.org/2021.emnlp-main.121.pdf) as inspiration. 




### Action Items: 
- Implement a POC with a small dataset and a small model
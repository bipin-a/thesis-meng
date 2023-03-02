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

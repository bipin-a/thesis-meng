# Adversarial Attacks (General Case)

## Robustness 
- A good model must be generalizable to unforeseen which is not divergent from training data in order to be robust.
- If the unforeseen data is divergent from the training data, it is a prompt to retrain due to a data shift.
- Adversarial attacks are a method of generating unforeseen data.
- If a model performs poorly on the adversarial dataset, it is a signal that the model has not learned the correct features from the data. This can be attributed to:
    1. Data not being diverse enough
    2. Model architecture or hyperparameters 

### Experiment 1: Is a poor performance of a model to an adversarial attack due to lack diverse dataset or poor hyperparameters and archecture? 
- Hypothesis: *Model X* has a robust archecture and hyperparams if after adversarial training the model has improved performance by `n`% .
- Underhood: By virture of adversarial training, a diverse dataset is generated. And the model can then perform significantly better.
- Challenge: Define the `n`% threshold.  

- Steps:
    1. Measure baseline evaluation metric value:
        - Input: Original Dataset (D), Model (M), 
        - Output: Evaluation Metric Value (V)
    2. Generate adversarially perturbed dataset:
        - Input: D, M, Attack Method (A)    
        - Output: Perturbed Dataset (D')
    3. Measure performance of original model on perturbed dataset 
        - Input: M, D'
        - Output: Evaluation Metric Value (m)
    4. Adversarial Training, Retrain* the M with D'  
        - Input: M, D'  
        - Output: Newly Fit Model (M')
    5. Measure performance of retrained model on perturbed dataset
        - Input: M', D'
        - Output: Evaluation Metric Value (m')

    **NOTE**: Not sure if step 4 should be retrain from scratch or transfer learned?

- Outcome: 
    1. Distribution of improvements from m to m' with:
        - Over 2-3 models
        - 2 adversarial attacks
        - 2-3 datasets. 

### Experiment 2: 

- Hypothesis: The perturbations of an adversarial attacks is dependant on the victim model (regardless of blackbox vs whitebox) fwiw all attacks are untargetted

- Steps:
    1. Select an adv attack 
    2. Select a dataset 
    3. Test a range of Victim Models
    4. Generate list of perturbed datasets for each victim model
    5. Compare overlap of inputs across lists datasets  
    6. Repeat steps 1-6 with different white box adv attacks 

- Results:  
    1. TextFoolerJin2019. Greedy attack with word importance ranking ["Is Bert Really Robust?" (Jin et al., 2019)](https://arxiv.org/abs/1907.11932):
        - Hypothesis is **correct**, perturbed datasets vary between the victim models
        - black-box model

    2. HotFlipEbrahimi2017. ["HotFlip: White-Box Adversarial Examples for Text Classification" (Ebrahimi et al., 2017)](https://arxiv.org/abs/1712.06751)


    3. BAEGarg2019. ["BAE: BERT-based Adversarial Examples for Text Classification" (Garg & Ramakrishnan, 2019](https://arxiv.org/abs/2004.01970).


# Adversarial Attacks on Language Models

## Transferability -  Understand the similarity of the features learned
- Generation of adversarial examples with high transferability allows for black-box attacks (train a substitute model and utilize the transferability of adversarial examples for the attack when they have no access and query restriction to target models)

### Experiment 1: Black Box attacks are transferable across all models with the same objective

- Hypothesis: Model A which has learned the same features from the text as Model B because Model A is a lite version of Model B, should be equally prone to an attack originally designed for Model B. 


- Steps:
    1. Measure baseline evaluation metric value:
        - Input: Original Dataset (D), Model_a, Model_b, Modelc (M_), 
        - Output: Evaluation Metric Value for each model (V_)
    2. Generate adversarially perturbed dataset:
        - Input: D, M, Attack Method (A)    
        - Output: Perturbed Dataset (D')
    3. Measure performance of original model on perturbed dataset 
        - Input: M, D'
        - Output: Evaluation Metric Value (m')
    




### Experiment 2: White box attacks should be transferable if it is `fit` on the same model.
- Hypothesis: Model A which has a encoder and decoder archecture vs Model B which has a transformer archecture are both equally prone to a blackbox attack for the opposite model.

## Linearity
- 

# Open Problems and Next Steps: 

## Innovation: 
1. Use 

2. Transform text to: 
    - Question and answering 
    - Text Summarization 

## Robustness

1. As we have emphasized in this paper, we recommend researchers and users to be EXTREMELY mindful on the quality of generated adversarial examples in natural language

2. We recommend the field to use human-evaluation derived thresholds for setting up constraints

3. Explor metrics such as Perplexity to quantify the quality of adverarial examples 
    - Note that the output value is based heavily on what text the model was trained on. This means that perplexity scores are not comparable between models or datasets.

    - In addition, perplexity is inapplicable to unnormalized language models (i.e., models that are not true probability distributions that sum to [eval-metrics-bntuw-9802.pdf](https://www.cs.cmu.edu/~roni/papers/eval-metrics-bntuw-9802.pdf))

    - https://huggingface.co/docs/transformers/perplexity


<figure align="center">
  <img 
    src="https://textattack.readthedocs.io/en/latest/_images/table5-main.png" 
    width="350" 
    alt="https://textattack.readthedocs.io/en/latest/_images/table5-main.png" 
    title="https://textattack.readthedocs.io/en/latest/_images/table5-main.png">
    <figcaption>Fig.1 - textattack.</figcaption>
</figure>

<figure align="center">
  <img 
    src="https://textattack.readthedocs.io/en/latest/_images/table3.png" 
    width="350" 
    alt="https://textattack.readthedocs.io/en/latest/_images/table3.png" 
    title="https://textattack.readthedocs.io/en/latest/_images/table3.png">
    <figcaption>Fig.1 - textattack.</figcaption>
</figure>

4. More compute and RAM pls 



##  Transferability 

- How to calculate the `Base transferability Rate`


## Linearity 

1. Are adversarial attacks linear? 
    - Linear in the dimension of:
        - vector embedding distance  
        - number of words transformed 



# Glossary: 

1. Link to Adversarial Attacks in NLP https://qdata.github.io/qdata-page/pic/20210414-HMI-textAttack.pdf

2. Big papers: https://github.com/thunlp/TAADpapers


## TextFoolerJin2019

| Attack Results                	| distilbert-base-uncased-imdb |albert-base-imdb |
|-------------------------------	|--------	|---------	|
| Number of successful attacks: 	| 4      	| 3       	|
| Number of failed attacks:     	| 0      	| 0       	|
| Number of skipped attacks:    	| 1      	| 2       	|
| Original accuracy:            	| 80.0%  	| 60.0%   	|
| Accuracy under attack:        	| 0.0%   	| 0.0%    	|
| Attack success rate:          	| 100.0% 	| 100.0%  	|
| Average perturbed word %:     	| 6.5%   	| 10.04%  	|
| Average num. words per input: 	| 215.6  	| 215.6   	|
| Avg num queries:              	| 546.75 	| 1112.67 	|




|    | original_text| perturbed_text_distilbert-base| perturbed_text_albert-base|
|---:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | <font color = red>I</font> love sci-fi and am willing to put up with a lot. Sci-fi <font color = red>movies</font>/TV are usually underfunded, under-appreciated and misunderstood. I <font color = red>tried</font> to like this, <font color = red>I</font> really did, but it is to good TV sci-fi as <font color = red>Babylon</font> 5 is to Star Trek (the <font color = red>original</font>). <font color = red>Silly</font> <font color = red>prosthetics</font>, cheap cardboard sets, <font color = red>stilted</font> dialogues, <font color = red>CG</font> that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and <font color = red>uninspiring</font>.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (<font color = red>cf</font>. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are <font color = red>wooden</font> and <font color = red>predictable</font>, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say "Gene Roddenberry's Earth..." otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this <font color = red>dull</font>, <font color = red>cheap</font>, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into <font color = red>space</font>. <font color = red>Spoiler</font>. So, kill off a main <font color = red>character</font>. And then bring him back as another <font color = red>actor</font>. Jeeez! Dallas all over again. | <font color = green>me</font> love sci-fi and am willing to put up with a lot. Sci-fi <font color = green>video</font>/TV are usually underfunded, under-appreciated and misunderstood. I <font color = green>deemed</font> to like this, <font color = green>me</font> really did, but it is to good TV sci-fi as <font color = green>Babel</font> 5 is to Star Trek (the <font color = green>prime</font>). <font color = green>Beast</font> <font color = green>dentures</font>, cheap cardboard sets, <font color = green>diction</font> dialogues, <font color = green>CJ</font> that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and <font color = green>distasteful</font>.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (<font color = green>fc</font>. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are <font color = green>bois</font> and <font color = green>predict</font>, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say "Gene Roddenberry's Earth..." otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this <font color = green>colourless</font>, <font color = green>tania</font>, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into <font color = green>distance</font>. <font color = green>Baffle</font>. So, kill off a main <font color = green>idiosyncrasies</font>. And then bring him back as another <font color = green>virtuoso</font>. Jeeez! Dallas all over again. | I <font color = green>darling</font> sci-fi and am willing to put up with a lot. Sci-fi movies/TV are <font color = green>periodically</font> underfunded, under-appreciated and misunderstood. I <font color = green>judged</font> to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the <font color = green>initial</font>). Silly prosthetics, cheap <font color = green>libretto</font> sets, <font color = green>creaky</font> dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi <font color = green>TELE</font>. It's not. It's clichéd and <font color = green>blunt</font>.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of <font color = green>subsistence</font>. Their actions and <font color = green>replied</font> are wooden and predictable, often <font color = green>tough</font> to watch. The makers of <font color = green>Terrain</font> <font color = green>REALISING</font> it's rubbish as they have to always say "Gene Roddenberry's Earth..." otherwise people would not continue watching. Roddenberry's <font color = green>cremated</font> must <font color = green>ser</font> turning in their orbit as this dull, cheap, poorly edited (watching it without advert <font color = green>skipping</font> really brings this home) trudging Trabant of a show lumbers into <font color = green>distance</font>. Spoiler. So, kill off a <font color = green>supreme</font> <font color = green>typeface</font>. And then bring him back as another actor. Jeeez! <font color = green>Houston</font> all over again. |
|  1 | Worth the entertainment value of a rental, especially if you like action <font color = red>movies</font>. This one features the usual car chases, fights with the great Van Damme kick style, shooting battles with the 40 shell load shotgun, and even terrorist style bombs. All of this is entertaining and competently <font color = red>handled</font> but there is nothing that really blows you away if you've seen your share before.<br /><br />The plot is made interesting by the inclusion of a rabbit, which is clever but <font color = red>hardly</font> profound. Many of the characters are heavily <font color = red>stereotyped</font> -- the angry veterans, the terrified illegal aliens, the crooked cops, the indifferent feds, the bitchy tough lady station head, the crooked politician, the fat federale who looks like he was typecast as the Mexican in a Hollywood movie from the 1940s. All passably <font color = red>acted</font> but again nothing special.<br /><br />I thought the main villains were pretty well done and fairly well acted. By the end of the movie you certainly knew who the good guys were and weren't. There was an emotional lift as the really bad ones got their just deserts. Very simplistic, but then you weren't expecting Hamlet, right? The only thing <font color = red>I</font> found <font color = red>really</font> <font color = red>annoying</font> was the constant cuts to VDs daughter during the last fight scene.<br /><br />Not bad. Not <font color = red>good</font>. Passable 4.| Worth the entertainment value of a rental, especially if you like action <font color = green>images</font>. This one features the usual car chases, fights with the great Van Damme kick style, shooting battles with the 40 shell load shotgun, and even terrorist style bombs. All of this is entertaining and competently <font color = green>transformed</font> but there is nothing that really blows you away if you've seen your share before.<br /><br />The plot is made interesting by the inclusion of a rabbit, which is clever but <font color = green>simply</font> profound. Many of the characters are heavily <font color = green>prejudices</font> -- the angry veterans, the terrified illegal aliens, the crooked cops, the indifferent feds, the bitchy tough lady station head, the crooked politician, the fat federale who looks like he was typecast as the Mexican in a Hollywood movie from the 1940s. All passably <font color = green>waged</font> but again nothing special.<br /><br />I thought the main villains were pretty well done and fairly well acted. By the end of the movie you certainly knew who the good guys were and weren't. There was an emotional lift as the really bad ones got their just deserts. Very simplistic, but then you weren't expecting Hamlet, right? The only thing <font color = green>me</font> found <font color = green>admittedly</font> <font color = green>infuriating</font> was the constant cuts to VDs daughter during the last fight scene.<br /><br />Not bad. Not <font color = green>super</font>. Passable 4.| Worth the entertainment value of a rental, especially if you like action movies. This one features the usual car chases, fights with the great Van Damme kick style, shooting battles with the 40 shell load shotgun, and even terrorist style bombs. All of this is entertaining and competently handled but there is nothing that really blows you away if you've seen your share before.<br /><br />The plot is made interesting by the inclusion of a rabbit, which is clever but hardly profound. Many of the characters are heavily stereotyped -- the angry veterans, the terrified illegal aliens, the crooked cops, the indifferent feds, the bitchy tough lady station head, the crooked politician, the fat federale who looks like he was typecast as the Mexican in a Hollywood movie from the 1940s. All passably acted but again nothing special.<br /><br />I thought the main villains were pretty well done and fairly well acted. By the end of the movie you certainly knew who the good guys were and weren't. There was an emotional lift as the really bad ones got their just deserts. Very simplistic, but then you weren't expecting Hamlet, right? The only thing I found really annoying was the constant cuts to VDs daughter during the last fight scene.<br /><br />Not bad. Not good. Passable 4.|


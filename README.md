# Natural Language Processing Course :

## Aspect Term Polarity Classification assignment

> NLP Assignment: Aspect-Term Polarity Classification in Sentiment Analysis.
> 
> DSBA Master – NLP Lecture 2024
> 
> Group: Antoine Debouchage, Juntao Liang, Pierre Prévot-Helloco, Clement Wang

## Requirement

* torch==2.1.0
* transformers==4.34.1
* tokenizers==0.14.1
* datasets==2.14.5
* scikit-learn==1.2.1
* numpy==1.26.0
* pandas==2.1.1

To install requirements, run `pip install -r requirements.txt`.

## Usage

### Training

Test for the assignment :

```sh
python -m src.tester
```

For personal training only:

```sh
python run.py
```

## Implementation and Architecture

The goal of this project is to classify between three labels (neutral / positive / negative) how a sentence is perceived given an aspect of it (wether it is about the food quality, the general ambiance and so on) highlighted by a specific word in the sentence.



![sample.png](images\sample.png)

The arhictecture that we used remains simple with a pretrained DistilBert model from which the pooled output are then concatenated with additional features from the aspect-term of the sample and fed to a multi-layer perceptron.

![model.png](images\model.png)





## BERT-based model

### distilbert-base-uncased ([official](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation))

Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv preprint arXiv:1910.01108 (2019). [[pdf](https://arxiv.org/pdf/1910.01108)]

## Licence

Centrale Supélec

# Natural Language Processing Course :

## Aspect Term Polarity Classification assignment

> NLP Assignment: Aspect-Term Polarity Classification in Sentiment Analysis.
> 
> DSBA Master – NLP Lecture 2024
> 
> Group: Antoine Debouchage, Juntao Liang, Pierre Prévot-Helloco, Clement Wang

![LICENSE](https://img.shields.io/badge/License-Centrale%20Sup%C3%A9lec-brightgreen)
![Guided By](https://img.shields.io/badge/Guided%20By-Naver%20Labs-cyan)

![ALL CONTRIBUTORS](https://img.shields.io/badge/All%20Contributors-4-orange)

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
* See [dataset.py](src/dataset.py) for augmentations.
* Check [classifier.py](src/classifier.py) for training arguments.
* Refer to [model.py](src/model.py) for aspect model implemented.

### Tips

* BERT-based models are sensitive to hyperparameters (especially learning rate) on small data sets.
* Fine-tuning on the specific task is necessary for releasing the true power of BERT.

## Implementation and Architecture

The goal of this project is to classify between three labels (neutral / positive / negative) how a sentence is perceived given an aspect of it (wether it is about the food quality, the general ambiance and so on) highlighted by a specific word in the sentence.

<img width="1366" alt="sample" src="https://github.com/antoine311200/nlp-aspect-term-polarity/assets/41137133/5a8898ae-93f8-43f0-be19-8d5109b15b5f">

The arhictecture that we used remains simple with a pretrained DistilBert model from which the pooled output are then concatenated with additional features from the aspect-term of the sample and fed to a multi-layer perceptron.

<img width="777" alt="model" src="https://github.com/antoine311200/nlp-aspect-term-polarity/assets/41137133/db3b91a5-d3fd-4a79-9c9e-c6d06c81098a">

## Reviews / Surveys

Qiu, Xipeng, et al. "Pre-trained models for natural language processing: A survey." Science China Technological Sciences 63.10 (2020): 1872-1897. [[pdf]](https://arxiv.org/pdf/2003.08271)

Adoma, Acheampong Francisca, Nunoo-Mensah Henry, and Wenyu Chen. "Comparative analyses of bert, roberta, distilbert, and xlnet for text-based emotion recognition." 2020 17th International Computer Conference on Wavelet Active Media Technology and Information Processing (ICCWAMTIP). IEEE, 2020. [[pdf]](https://arxiv.org/ftp/arxiv/papers/2104/2104.02041.pdf)

Young, Tom, et al. "Recent trends in deep learning based natural language processing." ieee Computational intelligenCe magazine 13.3 (2018): 55-75. [[pdf]](https://arxiv.org/pdf/1708.02709)

## BERT-based model

### distilbert-base-uncased ([model.py](src/model.py))([official](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation))

Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv preprint arXiv:1910.01108 (2019). [[pdf](https://arxiv.org/pdf/1910.01108)]

## Licence

Centrale Supélec

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

### Architecture

The architecture that we used remains simple with a pretrained DistilBert model from which the pooled output are then concatenated with additional features from the aspect-term of the sample and fed to a multi-layer perceptron.

The main idea was to find a way to tell the model where exactly was the aspect-term in the sentence and to give it also a hint about the category of the aspect-term directly in the input text. This was done by adding `[SEP]` tokens between the main aspect, the sub-theme and the sentence itself.
We later improved this by adding two specific tokens to delimit the aspect-term in the sentence, respectively `[START_WORD]` and `[END_WORD]`. This way, the model can learn to focus on the aspect-term and its context.

We further use a one-hot encoding of the aspect-term category and sub-category to enhance the model's understanding of the aspect-term and passing additional information to the model through a multi-layer perceptron that would concatenate the output of the DistilBert model with these categorical features.


<img width="777" alt="model" src="https://github.com/antoine311200/nlp-aspect-term-polarity/assets/41137133/db3b91a5-d3fd-4a79-9c9e-c6d06c81098a">


### Data augmentation

As the dataset is very small for a NLP task, we tried to augment the data by using a simple technique :
- get a list of every word aspect in the dataset for each category
- for each sentence, replace the aspect-term by another aspect-term of the same category

This way, we can increase the size of the dataset and pray that the model will generalize better. However, this technique is not perfect at all because the replacement is done randomly and the new sentence might not make sense at all making the DistilBert model learn from bad examples and failing to generalize.

## Results and Discussion

The model is trained on 1503 samples and tested on 376 samples. One comment that can be made regarding our results is that given the small size of the dataset, the model is prone to overfitting. This is why we have tried to add a dropout layer in the multi-layer perceptron to prevent this from happening. However, tby doing so we had to add another layer to the MLP to compensate which has complexified the model and made it as prone to overfitting as before.

Thus, we kept it simple and only used a one layer linear layer after the concatenation of the DistilBert output and the categorical features. The results are as follows:

On the training set:
| Metric | Value |
| --- | --- |
| Accuracy | 0.95 |
| F1 Score | 0.93 |
| Loss | 0.04 |

On the test set:
| Metric | Value |
| --- | --- |
| Accuracy | 0.80 |
| F1 Score | 0.79 |
| Loss | 0.08 |

This is the results that we get on average. Due to the length of the training, only one run was made for these results.

What could be done to improve the model would be to add more data to the training set, to add more layers to the MLP and to add more dropout layers to prevent overfitting. We could also try to use a different model such as Roberta or XLNet to see if the results are better.

## Conclusion

In conclusion, the model is able to predict the sentiment of a sentence given an aspect-term with a good accuracy and F1 score. However, the model is prone to overfitting and the results are not as good as we would have hoped. This is due to the small size of the dataset and the complexity of the task.
A proper ablation study would be necessary to understand the impact of each feature and hyperparameter on the model's performance. As well as a more thorough hyperparameter tuning to find the best configuration for the model.

## Reviews / Surveys

Qiu, Xipeng, et al. "Pre-trained models for natural language processing: A survey." Science China Technological Sciences 63.10 (2020): 1872-1897. [[pdf]](https://arxiv.org/pdf/2003.08271)

Adoma, Acheampong Francisca, Nunoo-Mensah Henry, and Wenyu Chen. "Comparative analyses of bert, roberta, distilbert, and xlnet for text-based emotion recognition." 2020 17th International Computer Conference on Wavelet Active Media Technology and Information Processing (ICCWAMTIP). IEEE, 2020. [[pdf]](https://arxiv.org/ftp/arxiv/papers/2104/2104.02041.pdf)

Young, Tom, et al. "Recent trends in deep learning based natural language processing." ieee Computational intelligenCe magazine 13.3 (2018): 55-75. [[pdf]](https://arxiv.org/pdf/1708.02709)

## BERT-based model

### distilbert-base-uncased ([model.py](src/model.py))([official](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation))

Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv preprint arXiv:1910.01108 (2019). [[pdf](https://arxiv.org/pdf/1910.01108)]

## License

CentraleSupélec

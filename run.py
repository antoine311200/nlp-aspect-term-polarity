import torch
from src.classifier import Classifier, Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()
classifier = Classifier(config)

datafile = "data/traindata.csv"
devfile = "data/devdata.csv"
classifier.train(datafile, devfile, device=device)
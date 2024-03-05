from typing import List

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from src.model import AspectModel
from src.dataset import AspectDataset

from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int = 32
    learning_rate: float = 3e-3
    num_epochs: int = 5
    num_labels: int = 3
    model_name: str = "distilbert-base-uncased"


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
    """

    ############################################# complete the classifier class below

    def __init__(self, config=None):
        """
        This should create and initilize the model. Does not take any arguments.

        """
        self.config = config
        self.model_name = config.model_name
        self.num_labels = config.num_labels

        print(f"Using model: {self.model_name}")
        print(f"Number of labels: {self.num_labels}")

        self.model = AspectModel(self.num_labels, self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        self.device = device

        train_dataset = AspectDataset(train_filename, self.tokenizer)
        dev_dataset = AspectDataset(dev_filename, self.tokenizer)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        dev_loader = DataLoader(dev_dataset, batch_size=self.config.batch_size)

        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.num_epochs):
            self.model.train()
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print('Training:')
            for batch in train_loader:
                self.train_step(batch, optimizer)

            print('Validation:')
            for batch in dev_loader:
                self.validation_step(batch)

    def train_step(self, batch, optimizer):
        input_ids       = batch["input_ids"].to(self.device)
        attention_mask  = batch["attention_mask"].to(self.device)
        theme           = batch["theme"].to(self.device)
        subtheme        = batch["subtheme"].to(self.device)
        start_word      = batch["start_word"].to(self.device)
        end_word        = batch["end_word"].to(self.device)
        label           = batch["label"].to(self.device)

        optimizer.zero_grad()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            theme=theme,
            subtheme=subtheme,
            start_word=start_word,
            end_word=end_word,
            label=label
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

    def validation_step(self, batch):
        input_ids       = batch["input_ids"].to(self.device)
        attention_mask  = batch["attention_mask"].to(self.device)
        theme           = batch["theme"].to(self.device)
        subtheme        = batch["subtheme"].to(self.device)
        start_word      = batch["start_word"].to(self.device)
        end_word        = batch["end_word"].to(self.device)
        label           = batch["label"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            theme=theme,
            subtheme=subtheme,
            start_word=start_word,
            end_word=end_word,
            label=label
        )

        loss = outputs.loss
        accuracy = (outputs.logits.argmax(dim=1) == label).float().mean()

        print(f"Validation Loss: {loss.item()}")
        print(f"Validation Accuracy: {accuracy.item()}")

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """

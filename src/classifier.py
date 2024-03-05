from typing import List

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.model import AspectModel
from src.dataset import AspectDataset

from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int = 32
    learning_rate: float = 5e-3 # 3e-3
    num_epochs: int = 10
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

        num_train_steps = int(len(train_dataset) * self.config.num_epochs / self.config.batch_size)
        print(f"Number of training steps: {num_train_steps}")
        num_warmup_steps = int(num_train_steps * 0.05)
        print(f"Number of warmup steps: {num_warmup_steps}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
        )

        print(f"Training model for {self.config.num_epochs} epochs on device {device}")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")

            losses = []
            accuracies = []

            print('Training:')
            for batch in train_loader:
                loss, accuracy = self.train_step(batch, optimizer, scheduler)
                losses.append(loss.item())
                accuracies.append(accuracy.item())

                avg_loss = sum(losses)/len(losses)
                avg_accuracy = sum(accuracies)/len(accuracies)
                print(f"  loss: {avg_loss:.2f} | accuracy: {avg_accuracy:.2f}", end='\r')

            print(f"  epoch loss: {avg_loss:.2f} | epoch accuracy: {avg_accuracy:.2f}")

            val_losses = []
            val_accuracies = []

            print('Validation:')
            self.model.eval()
            for batch in dev_loader:
                loss, accuracy = self.validation_step(batch)

                val_losses.append(loss.item())
                val_accuracies.append(accuracy.item())

            avg_loss = sum(val_losses)/len(val_losses)
            avg_accuracy = sum(val_accuracies)/len(val_accuracies)
            print(f"  loss: {avg_loss:.2f} | accuracy: {avg_accuracy:.2f}")

        print(f"  validation loss: {avg_loss:.2f} | validation accuracy: {avg_accuracy:.2f}")

        # Save the model
        print("Finished training. Saving model...")
        torch.save(self.model.state_dict(), "aspect_model.pth")

    def train_step(self, batch, optimizer, scheduler):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)

        loss = outputs['loss']
        accuracy = (outputs['logits'].argmax(dim=1) == batch['label']).float().mean()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        return loss, accuracy

    def validation_step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)

        loss = outputs['loss']
        accuracy = (outputs['logits'].argmax(dim=1) == batch['label']).float().mean()

        return loss, accuracy

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """

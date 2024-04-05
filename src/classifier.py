from typing import List
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score as f1_scorer

from model import AspectModel
from dataset import AspectDataset



@dataclass
class Config:
    batch_size: int = 16
    learning_rate: float = 1e-4 # 3e-3
    num_epochs: int = 10
    num_labels: int = 3
    model_name: str = "distilbert-base-uncased"
    scheduler: str = "cosine"
    use_class_weights: bool = True
    use_augmentation: bool = False


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
        self.config = config if config is not None else Config()
        self.model_name = self.config.model_name
        self.num_labels = self.config.num_labels

        print(f"Using model: {self.model_name}")
        print(f"Number of labels: {self.num_labels}")

        self.model = AspectModel(self.num_labels, self.model_name, self.config.use_class_weights)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.theme_token = "[THEME]"
        self.tokenizer.subtheme_token = "[SUBTHEME]"
        self.tokenizer.start_word_token = "[START_WORD]"
        self.tokenizer.end_word_token = "[END_WORD]"
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [self.tokenizer.theme_token, self.tokenizer.subtheme_token, self.tokenizer.start_word_token, self.tokenizer.end_word_token]
        })
        self.model.distilbert.resize_token_embeddings(len(self.tokenizer))

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

        train_dataset = AspectDataset(train_filename, self.tokenizer, use_augmentation=self.config.use_augmentation)
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

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-2)
        scheduler = self.get_scheduler(optimizer, num_warmup_steps, num_train_steps)

        print(f"Training model for {self.config.num_epochs} epochs on device {device}")

        for epoch in range(self.config.num_epochs):
            self.model.train()
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")

            losses = []
            accuracies = []
            f1_scores = []

            print('Training:')
            for batch in train_loader:
                loss, accuracy, f1_score = self.train_step(batch, optimizer, scheduler, class_weights=train_dataset.class_weights)
                losses.append(loss.item())
                accuracies.append(accuracy.item())
                f1_scores.append(f1_score)

                avg_loss = sum(losses)/len(losses)
                avg_accuracy = sum(accuracies)/len(accuracies)
                avg_f1_score = sum(f1_scores)/len(f1_scores)

                print(f"  loss: {avg_loss:.2f} | accuracy: {avg_accuracy:.2f} | f1 score: {avg_f1_score:.2f}", end='\r')

            print(f"  epoch loss: {avg_loss:.2f} | epoch accuracy: {avg_accuracy:.2f} | epoch f1 score: {avg_f1_score:.2f}")

            val_losses = []
            val_accuracies = []
            val_f1_scores = []

            print('Validation:')
            self.model.eval()
            for batch in dev_loader:
                loss, accuracy, f1_score = self.validation_step(batch, class_weights=dev_dataset.class_weights)

                val_losses.append(loss.item())
                val_accuracies.append(accuracy.item())
                val_f1_scores.append(f1_score)

            avg_loss = sum(val_losses)/len(val_losses)
            avg_accuracy = sum(val_accuracies)/len(val_accuracies)
            avg_f1_score = sum(val_f1_scores)/len(val_f1_scores)
            print(f"  loss: {avg_loss:.2f} | accuracy: {avg_accuracy:.2f} | f1 score: {avg_f1_score:.2f}")

        print(f"  validation loss: {avg_loss:.2f} | validation accuracy: {avg_accuracy:.2f}")

        # Save the model
        print("Finished training. Saving model...")
        torch.save(self.model.state_dict(), "aspect_model.pth")

    def train_step(self, batch, optimizer, scheduler, class_weights=None):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch['class_weights'] = class_weights
        outputs = self.model(**batch)

        loss = outputs['loss']
        accuracy = (outputs['logits'].argmax(dim=1) == batch['label']).float().mean()
        f1_score = f1_scorer(batch['label'].cpu(), outputs['logits'].argmax(dim=1).cpu(), average='weighted')

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        return loss, accuracy, f1_score

    def validation_step(self, batch, class_weights=None):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch['class_weights'] = class_weights
        outputs = self.model(**batch)

        loss = outputs['loss']
        accuracy = (outputs['logits'].argmax(dim=1) == batch['label']).float().mean()
        f1_score = f1_scorer(batch['label'].cpu(), outputs['logits'].argmax(dim=1).cpu(), average='weighted')

        return loss, accuracy, f1_score

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        dataset = AspectDataset(data_filename, self.tokenizer)
        loader = DataLoader(dataset, batch_size=self.config.batch_size)

        self.model.to(device)
        self.model.eval()


        predictions = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.model(**batch)

            predictions.extend(outputs['logits'].argmax(dim=1).cpu().numpy())

        reverse_label = {0: 'neutral', 1: 'positive', 2: 'negative'}
        
        return [reverse_label[pred] for pred in predictions]



    def get_scheduler(self, optimizer, num_warmup_steps, num_training_steps):
        if self.config.scheduler == "linear":
            return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        elif self.config.scheduler == "cosine":
            return CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=0.0)
        elif self.config.scheduler == "none":
            return None
        else:
            raise ValueError(f"Invalid scheduler: {self.config.scheduler}")
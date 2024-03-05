import torch
import torch.nn as nn
from transformers import (
    DistilBertForSequenceClassification,
    AutoConfig,
)


class AspectModel(DistilBertForSequenceClassification):

    def __init__(self, num_labels, model_name="distilbert-base-uncased"):
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        super(AspectModel, self).__init__(config)
        self.num_labels = num_labels
        self.model_name = model_name

        # Hard coded dimensions for now
        self.text_feat_dim = 768  # DistilBert hidden size
        self.cat_feat_dim = 4  # 4 categories: theme, subtheme, start_word, end_word
        self.total_feat_dim = self.text_feat_dim + self.cat_feat_dim

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)

        # MLP for classification on top of the concatenated features
        # This will need to be modified up to avoiding overfitting
        self.mlp = nn.Sequential(
            nn.Linear(self.total_feat_dim, 16),
            nn.ReLU(),
            self.dropout,
            nn.Linear(16, self.num_labels),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        theme,
        subtheme,
        start_word,
        end_word,
        label,
        class_weights=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=False,
        return_dict=None,
    ):
        # Get the text features from the DistilBert model
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_features = outputs['last_hidden_state'][:, 0, :] # Get the CLS embedding
        text_features = self.dropout(text_features)

        # Concatenate additional features with text features
        additional_features = torch.cat([theme.unsqueeze(1), subtheme.unsqueeze(1), start_word.unsqueeze(1), end_word.unsqueeze(1)], dim=-1)
        all_features = torch.cat((text_features, additional_features), dim=-1)

        # Get the logits and probabilities from the classifier
        logits = self.mlp(all_features)
        probs = self.softmax(logits)

        loss = None
        if label is not None:
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "probs": probs,
        }


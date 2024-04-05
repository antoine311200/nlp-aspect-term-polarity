import torch
import torch.nn as nn
from transformers import (
    DistilBertForSequenceClassification,
    AutoConfig,
)


class AspectModel(DistilBertForSequenceClassification):

    def __init__(self, num_labels, model_name="distilbert-base-uncased", use_class_weights=False):
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        super(AspectModel, self).__init__(config)
        self.num_labels = num_labels
        self.model_name = model_name
        self.use_class_weights = use_class_weights

        # Hard coded dimensions for now
        self.text_feat_dim = 768  # DistilBert hidden size
        self.cat_feat_dim = 11 # 6 for theme, 5 for subtheme
        self.total_feat_dim = self.text_feat_dim + self.cat_feat_dim

        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

        # MLP for classification on top of the concatenated features
        # This will need to be modified up to avoiding overfitting
        self.mlp = nn.Sequential(
            nn.Linear(self.total_feat_dim, 16),
            nn.ReLU(),
            self.dropout,
            # nn.Linear(64, 16),
            # nn.ReLU(),
            # self.dropout,
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
        pooled_output = outputs['last_hidden_state'][:, 0, :] # Get the CLS embedding
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        text_features = self.dropout(pooled_output)

        # Concatenate additional features with text features
        theme = torch.nn.functional.one_hot(theme.long(), num_classes=6).float()
        subtheme = torch.nn.functional.one_hot(subtheme.long(), num_classes=5).float()

        additional_features = torch.cat([theme, subtheme], dim=-1)
        all_features = torch.cat((text_features, additional_features), dim=-1)

        # Get the logits and probabilities from the classifier
        logits = self.mlp(all_features)
        probs = self.softmax(logits)

        loss = None
        if label is not None:
            if not self.use_class_weights:
                    class_weights = None
                    
            if class_weights is not None:
                class_weights = class_weights.to(logits.device)
                class_weights = class_weights.float()
                

            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "probs": probs,
        }


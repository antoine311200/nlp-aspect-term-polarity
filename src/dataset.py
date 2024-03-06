import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import pandas as pd

class AspectDataset(Dataset):
    def __init__(self, datafile, tokenizer, max_length=256):

        # Tokenize the sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

        # sep token from the tokenizer
        self.sep_token = tokenizer.sep_token

        # Load the data from the file
        self.df = self._preprocess(datafile)
        self.df['input_ids'], self.df['attention_mask'] = zip(*self.df.apply(self._tokenize, axis=1))

        self.class_weights = torch.tensor(compute_class_weight('balanced', classes=self.df['label'].unique(), y=self.df['label']))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.df['input_ids'][idx]),
            'attention_mask': torch.tensor(self.df['attention_mask'][idx]),
            'label': torch.tensor(self.df['label'][idx]),
            'theme': torch.tensor(self.df['theme_encoded'][idx]),
            'subtheme': torch.tensor(self.df['subtheme_encoded'][idx]),
            'start_word': torch.tensor(self.df['start_word'][idx]),
            'end_word': torch.tensor(self.df['end_word'][idx]),
        }

    def _tokenize(self, row):
        result = self.tokenizer.encode_plus(
            row['word'] + self.sep_token +
            row['theme'] + self.sep_token +
            row['subtheme'] + self.sep_token +
            row['sentence'],

            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
        )
        return result['input_ids'], result['attention_mask']

    @staticmethod
    def _preprocess(datafile):
        column_names = ['label', 'aspect', 'word', 'pos', 'sentence']
        df = pd.read_csv(datafile, sep='\t', header=None, names=column_names)

        # Process the 'pos' column into two integer columns 'start_word' and 'end_word'
        df['start_word'] = df['pos'].apply(lambda x: int(x.split(':')[0]))
        df['end_word'] = df['pos'].apply(lambda x: int(x.split(':')[1]))

        # Process the 'label' column into a list of integers
        label_enum = {
            "neutral": 0,
            "positive": 1,
            "negative": 2
        }
        df['label'] = df['label'].apply(lambda x: label_enum[x])
        df['theme'] = df['aspect'].apply(lambda x: x.split('#')[0])
        df['subtheme'] = df['aspect'].apply(lambda x: x.split('#')[1])

        le = LabelEncoder()
        df['theme_encoded'] = le.fit_transform(df['theme'])
        df['subtheme_encoded'] = le.fit_transform(df['subtheme'])

        return df
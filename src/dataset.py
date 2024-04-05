import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.utils.class_weight import compute_class_weight

import pandas as pd
import random

class AspectDataset(Dataset):
    def __init__(self, datafile, tokenizer, max_length=256, use_augmentation=False):

        # Tokenize the sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

        # sep token from the tokenizer
        self.sep_token = tokenizer.sep_token

        self.theme_token = self.sep_token#tokenizer.theme_token
        self.subtheme_token = self.sep_token#tokenizer.subtheme_token
        self.start_word_token = tokenizer.start_word_token
        self.end_word_token = tokenizer.end_word_token

        self.special_tokens = tokenizer.special_tokens_map

        # Load the data from the file
        self.df = self._preprocess(datafile)
        self.df['input_ids'], self.df['attention_mask'] = zip(*self.df.apply(self._tokenize, axis=1))

        self.class_weights = torch.tensor(compute_class_weight('balanced', classes=self.df['label'].unique(), y=self.df['label']))

        self.use_augmentation = use_augmentation
        if self.use_augmentation:
            self.theme_words = {}
            for theme in self.df['theme'].unique():
                self.theme_words[theme] = self.df[self.df['theme'] == theme]['word'].unique()

    def __len__(self):
        return len(self.df) * 2 if self.use_augmentation else len(self.df)

    def __getitem__(self, idx):
        if self.use_augmentation and idx >= len(self.df):
            return self.get_augmentation(idx - len(self.df))
        
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

        sentence = row['sentence']

        sentence = sentence[:row['start_word']] + self.start_word_token  +" "+ sentence[row['start_word']:row['end_word']] + " " + self.end_word_token + sentence[row['end_word']:]
        result = self.tokenizer.encode_plus(
            row['word'] + self.theme_token +
            row['theme'] + self.subtheme_token +
            row['subtheme'] + self.sep_token +
            sentence,

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

        
        theme_enum = {
            "FOOD": 0,
            "SERVICE": 1,
            "AMBIENCE": 2,
            "RESTAURANT": 3,
            "DRINKS": 4,
            "LOCATION": 5,
        }
        subtheme_enum = {
            "QUALITY": 0,
            "PRICES": 1,
            "STYLE_OPTIONS": 2,
            "GENERAL": 3,
            "MISCELLANEOUS": 4,
        }

        df['theme_encoded'] = df['theme'].apply(lambda x: theme_enum[x])
        df['subtheme_encoded'] = df['subtheme'].apply(lambda x: subtheme_enum[x])

        return df
    
    def get_augmentation(self, idx):
        row = self.df.iloc[idx]
        candidate_words = self.theme_words[row["theme"]]
        new_word = candidate_words[random.randint(0, len(candidate_words)-1)]
        new_row = row.copy()

        new_row["word"] = new_word
        new_row["end_word"] = new_row["start_word"] + len(new_word)
        new_row["sentence"] = row["sentence"][:row["start_word"]] + new_word + row["sentence"][row["end_word"]:]
        input_ids, attention_mask = self._tokenize(new_row)

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'label': torch.tensor(new_row['label']),
            'theme': torch.tensor(new_row['theme_encoded']),
            'subtheme': torch.tensor(new_row['subtheme_encoded']),
            'start_word': torch.tensor(new_row['start_word']),
            'end_word': torch.tensor(new_row['end_word']),
        }
    
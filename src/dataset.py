import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder

import pandas as pd

class AspectDataset(Dataset):
    def __init__(self, datafile, tokenizer, max_length):
        # Load the data from the file
        self.df = self.__preprocess(datafile)

        # Tokenize the sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

        # sep token from the tokenizer
        self.sep_token = tokenizer.sep_token

        # Get the input IDs and attention masks
        self.input_ids = []
        self.attention_masks = []

        for i in range(len(self.df)):
            # Tokenize the word [sep] theme [sep] subtheme [sep] sentence
            tokens = self.tokenizer.encode_plus(
                self.df.loc[i, 'word'],
                self.df.loc[i, 'theme'],
                self.df.loc[i, 'subtheme'],
                self.df.loc[i, 'sentence'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )


    def __preprocess(datafile):
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
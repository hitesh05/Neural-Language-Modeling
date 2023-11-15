import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CSVDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length, subset_size = None):
        self.dataframe = pd.read_csv(filename)
        if subset_size is not None:
            self.dataframe = self.dataframe.sample(n = subset_size, random_state = 42).reset_index(drop= True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        article_text = self.dataframe.iloc[idx]['article']
        highlights_text = self.dataframe.iloc[idx]['highlights']

        # Encode the inputs and targets
        article_encoding = self.tokenizer.encode_plus(
            article_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        highlights_encoding = self.tokenizer.encode_plus(
            highlights_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': article_encoding['input_ids'].flatten(),
            'attention_mask': article_encoding['attention_mask'].flatten(),
            'labels': highlights_encoding['input_ids'].flatten()
        }
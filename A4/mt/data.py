import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# train_file_en = '/media/hitesh/DATA/IIIT-H/4th_year/Anlp/Assignments/A4/de-en/europarl-v7.de-en.en'
# train_file_de = '/media/hitesh/DATA/IIIT-H/4th_year/Anlp/Assignments/A4/de-en/europarl-v7.de-en.de'

class CSVDataset(Dataset):
    def __init__(self, file1, file2, tokenizer, max_length, subset_size = None):
        with open(file1, 'r') as f:
            self.en_lines = f.readlines()
        with open(file2, 'r') as f:
            self.de_lines = f.readlines()
        if subset_size is not None:
            self.en_lines = self.en_lines[:subset_size]
            self.de_lines = self.de_lines[:subset_size]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.en_lines)

    def __getitem__(self, idx):
        # Encode the inputs and targets
        en_encoding = self.tokenizer.encode_plus(
            self.en_lines[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        de_encoding = self.tokenizer.encode_plus(
            self.de_lines[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': en_encoding['input_ids'].flatten(),
            'attention_mask': en_encoding['attention_mask'].flatten(),
            'labels': de_encoding['input_ids'].flatten()
        }
import json
import pandas as pd
from torch.utils.data import Dataset

file = 'dataset/train.json'
with open(file, 'r') as f:
    data = json.load(f)['data']

paragraphs = self.data[idx]['paragraphs']
questions = [qa['question'] for para in paragraphs for qa in para['qas']]
# class CSVDataset(Dataset):
#     def __init__(self, json_file, tokenizer, max_length, subset_size=None):
#         with open(json_file, 'r') as f:
#             self.data = json.load(f)['data']
#         if subset_size is not None:
#             self.data = self.data[:subset_size]
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         paragraphs = self.data[idx]['paragraphs']
#         questions = [qa['question'] for para in paragraphs for qa in para['qas']]
#         answers = [qa['answers'][0]['text'] for para in paragraphs for qa in para['qas']]

#         # Encode the inputs and targets
#         question_encoding = self.tokenizer(questions,
#                                            add_special_tokens=True,
#                                            max_length=self.max_length,
#                                            return_attention_mask=True,
#                                            padding='max_length',
#                                            truncation=True,
#                                            return_tensors='pt')

#         answer_encoding = self.tokenizer(answers,
#                                          add_special_tokens=True,
#                                          max_length=self.max_length,
#                                          return_attention_mask=True,
#                                          padding='max_length',
#                                          truncation=True,
#                                          return_tensors='pt')

#         return {
#             'input_ids': question_encoding['input_ids'].flatten(),
#             'attention_mask': question_encoding['attention_mask'].flatten(),
#             'labels': answer_encoding['input_ids'].flatten()
#         }

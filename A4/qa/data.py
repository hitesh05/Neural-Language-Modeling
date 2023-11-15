import json
import pandas as pd
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length , fraction = 0.1):
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.answers = []

        # Load and parse the dataset
        with open(filename, 'r') as file:
            data = json.load(file)
        
        num_articles = int(len(data['data'])*fraction)
        data['data'] = data['data'][:num_articles]
            

        for article in data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    answer = qa['answers'][0]['text'] if not qa['is_impossible'] else '[NO ANSWER]'

                    # Format the input
                    input_text = f"Context: {context} Question: {question} Answer:"
                    self.inputs.append(input_text)
                    self.answers.append(answer)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        answer_text = self.answers[idx]
        input_text = f"[QUESTION] {input_text} [ANSWER]"

        # Encode the inputs and answers
        input_encoding = self.tokenizer.encode_plus(input_text, add_special_tokens=True, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        answer_encoding = self.tokenizer.encode_plus(answer_text, add_special_tokens=True, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)

        # Return as a dictionary
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': answer_encoding['input_ids'].flatten()
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data import *
import math

class My_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, h1=300, embedding_dim=300):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).from_pretrained(embedding_matrix)
        self.lstm = nn.LSTM(embedding_dim, h1, num_layers=2, batch_first=True)
        self.fc = nn.Linear(h1, vocab_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
    def forward(self, batch):
        x = self.embedding(batch)
        x,_ = self.lstm(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=2)
        return x
    
    def train(self, train_dataset, dev_dataset, num_epochs=10, lr=0.01):
        self.loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.SGD(self.parameters(), lr=lr)
        train_data = DataLoader(train_dataset, batch_size=256, shuffle=True)
        dev_data = DataLoader(dev_dataset, batch_size=256, shuffle=True)
        
        for i in range(num_epochs):
            for batch in tqdm(train_data, desc="training batch"):
                optim.zero_grad()
                contexts, words = batch
                contexts = contexts.to(self.device)
                words = words.to(self.device)
                outs = self.forward(contexts)
                outs = outs.view(-1, outs.shape[-1])
                words = words.view(-1)
                loss = self.loss_fn(outs, words)
                loss.backward()
                optim.step()
                
            dev_loss = self.get_test_loss(dev_data)
            dev_perp = self.get_perplexity(dev_data)
            print(f"Epoch {i+1} | Dev Loss: {dev_loss} | Dev Perplexity: {dev_perp}")
                
    def get_test_loss(self, test_dataset):
        loss = 0
        for batch in tqdm(test_dataset, desc="getting loss for test"):
            contexts, words = batch
            contexts = contexts.to(self.device)
            words = words.to(self.device)
            outs = self.forward(contexts)
            outs = outs.view(-1, outs.shape[-1])
            words = words.view(-1)
            l = self.loss_fn(outs, words)
            loss+=l.item()
        return loss
    
    def get_perplexity(self, data):
        perp = 0
        count = 0
        for batch in tqdm(data, desc='getting perplexity'):
            contexts, words = batch
            contexts = contexts.to(self.device)
            words = words.to(self.device)
            outs = self.forward(contexts)
            outs = outs.view(-1, outs.shape[-1])
            words = words.view(-1)
            loss = self.loss_fn(outs, words)
            loss = loss.item()
            prp = math.exp(loss)
            perp += prp
            count += len(batch)
        return (perp/count)
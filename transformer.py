import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data import *
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, max_seq_length):
        super(TransformerDecoder, self).__init__()
        self.d_model = embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=max_seq_length)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=self.d_model, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(self.d_model, vocab_size)

    def forward(self, input):
        embedded = self.embedding(input) * math.sqrt(self.d_model)
        
        
        ################NOTE###################
        ### take care of dimensions while passing, possible mistake here ###
        #####################################
        embedded = self.positional_encoding(embedded)
        
        
        x = embedded.permute(1, 0, 2)  # Adjust the dimensions for Transformer
        for layer in self.transformer_layers:
            x = layer(x)
        output = self.fc(x)
        return output
    
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
            perp += math.exp(loss.item())
            count += len(batch)
        return (perp/count)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data import *
import math

class NNLM(nn.Module):
    def __init__(self, vocab_size, context_size=5, h1=300, h2=300):
        super().__init__()
        self.layer1 = nn.Linear(context_size * vocab_size, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3 = nn.Linear(h2, vocab_size)
        
    def forward(self, batch):
        inputs = batch.view(batch.shape[0], -1)
        x = self.layer1(inputs)
        x = nn.ReLU(x)
        x = self.layer2(x)
        x = nn.ReLU(x)
        x = self.layer3(x)
        logits = nn.ReLU(x)
        return logits
    
    def train(self, train_dataset, dev_dataset, num_epochs=10, lr=0.01):
        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.SGD(self.parameters(), lr=lr)
        train_data = DataLoader(train_dataset, batch_size=256, shuffle=True)
        dev_data = DataLoader(dev_dataset, batch_size=256, shuffle=True)
        
        for i in range(num_epochs):
            for batch in tqdm(train_data, desc="training batch"):
                optim.zero_grad()
                contexts, words = batch
                outs = self.forward(contexts)
                loss = loss_fn(outs, words)
                loss.backward()
                optim.step()
                
            dev_loss = self.get_test_loss(dev_data, loss_fn)
            dev_perp = self.get_perplexity(loss_fn, dev_data)
            print(f"Epoch {i+1} | Dev Loss: {dev_loss} | Dev Perplexity: {dev_perp}")
                
    def get_test_loss(self, test_dataset, loss_fn):
        loss = 0
        for batch in tqdm(test_dataset, desc="getting loss for test"):
            contexts, words = batch
            outs = self.forward(contexts)
            l = loss_fn(outs, words)
            loss+=l.item()
        return loss
    
    def get_perplexity(self,loss_fn, data):
        perp = 0
        count = 0
        for batch in tqdm(data, desc='getting perplexity'):
            contexts, words = batch
            outs = self.forward(contexts)
            loss = loss_fn(outs, words)
            perp += math.exp(loss.item())
            count += len(batch)
        return (perp/count)
        
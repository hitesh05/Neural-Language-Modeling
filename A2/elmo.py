import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data import *
import math

class Elmo(nn.Module):
    def __init__(self, vocabulary, embedding_matrix, hsize=300):
        super().__init__()
        self.num_layers = 2
        self.hidden_size = hsize
        self.vocab = vocabulary
        
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=vocabulary['<pad>'])
        
        self.lstm = nn.LSTM(input_size = self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True)
        self.weights = torch.rand(self.num_layers)
        
        self.lm_head = nn.Linear(in_features=self.hidden_size*2, out_features=len(self.vocab))
        self.classification_head = nn.Linear(in_features=self.hidden_size*4, out_features=4)
        
    def forward(self, batch):
        # tokens/ctx: batch[0], labels/words: batch[1]
        embeddings = self.embedding(batch[0])
        
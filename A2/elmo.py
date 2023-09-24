import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

class Elmo(nn.Module):
    def __init__(self, vocabulary, embedding_matrix, hsize=300):
        super().__init__()
        self.hidden_size = hsize

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True, padding_idx=vocabulary['<PAD>'])

        # declaring 4 different lstms for a stacked (2-layer) bi-lstm
        self.lstm_f1 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.lstm_b1 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.lstm_f2 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.lstm_b2 = nn.LSTM(self.hidden_size, self.hidden_size)
        self.cls = nn.Linear(self.hidden_size, len(vocabulary))


    def forward(self, input_f, input_b):
        # getting fwd and bwd embeddings
        embeddings_f = self.embedding(input_f)
        embeddings_b = self.embedding(input_b)

        # getting fwd and bwd outputs from lstms in the bottom layer
        # and concatenating the output
        out_f1, _ = self.lstm_f1(embeddings_f)
        out_b1, _ = self.lstm_b1(embeddings_b)

        # same step for second layer of lstms
        out_f2,_ = self.lstm_f2(out_f1)
        out_b2,_ = self.lstm_b2(out_b1)
        pf = self.cls(out_f2)
        pb = self.cls(out_b2)

        return pf,pb
    
class Elmo_classifier(nn.Module):
    def __init__(self, model_path, embedding_matrix, hsize=300):
        self.hidden_size = hsize
        # declaring and loading pretrained elmo
        self.elmo = Elmo(embedding_matrix)
        self.elmo.load_state_dict(torch.load(model_path))
        
        # params for the weights (weighted mean of the layers)
        self.params = nn.Parameter(torch.empty(3))
        self.elmo_params.data.uniform_(0.5, 1.5)
        
        # lstm through which the embeddings obtained from elmo are passed
        self.lstm = nn.LSTM(2 * self.hidden_size, self.hidden_size)
        
        # declaring classification head for finetunig task
        self.cls_head = nn.Linear(self.hidden_size, 5)
        
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x, x_f, x_b):
        ### CHECK IF CORRECT ### 
        with torch.no_grad():
            # concatenating to make dims same
            w0 = torch.cat([x,x], dim=1)
            # getting contextualised embeddings from pretrained elmo
            w1, w2 = self.elmo(x_f, x_b)
        
        weights = self.softmax(self.params)
        # weighted mean of pretrained embeds and 2 contextualised embeds (from the 2 layers of the lstm)
        w = weights[0]*w0 + weights[1]*w1 + weights[2]*w2
        
        # passing thru lstm
        o ,(c,h)= self.lstm(w)
        
        # classifying
        return self.cls_head(torch.mean(o, dim=0))
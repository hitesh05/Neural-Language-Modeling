import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torchtext import vocab
from torch.utils.data import Dataset, DataLoader
import os
import random
import regex as re
from collections import defaultdict
import sys
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import math
import pandas as pd
from elmo import Elmo
from elmo_finetune import Elmo_classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProcessDataset():
    def __init__(self, filepath, min_freq=1):
        self.specials = ['<UNK>', '<PAD>']
        df = pd.read_csv(filepath)
        all_words = list()
        for i in tqdm(range(len(df)), desc="building vocabulary"):
            if i < 7600:
                continue
            if i > 8600:
                break
            line = df['Description'][i]
            l = [[word.lower()] for word in word_tokenize(line)]
            all_words.extend(l)

        self.vocab = vocab.build_vocab_from_iterator(all_words, min_freq=min_freq, specials=self.specials)
        def_idx = self.vocab[self.specials[0]]
        self.vocab.set_default_index(def_idx)

class ElmoDataset(Dataset):
    def __init__(self, vocabulary, filename, is_train=True):
        self.vocab = vocabulary
        self.ctx_f = list() # forward context for fwd lstm
        self.words_f = list() # fwd targets
        self.ctx_b = list() # backward context for bwd lstm
        self.words_b = list() # bwd targets
        self.max_len = -1 # max len for padding
        self.df = pd.read_csv(filename)
        self.specials = ['<UNK>', '<PAD>']
        self.is_train = is_train
        self.make_dataset()

    def make_dataset(self):
        for i in tqdm(range(len(self.df)), desc='tokenising'):
            # first 7600 sents of train as dev
            # only take around 34k sents for train to avoid running out of memory
            # take test = dev set = 7600 sents
            if self.is_train:
                if i < 7600:
                    continue
                if i > 8600:
                    break
            else:
                if i > 7600:
                    break

            # iterate over the dataset
            line = self.df['Description'][i]
            words = list()
            indices = list()
            for word in word_tokenize(line):
                words.append(word.lower())
            for word in words:
                indices.append(self.vocab[word])

            # making context and target for fwd for lstm
            self.ctx_f += list(indices[:i] for i in range(1, len(indices)))
            self.words_f += indices[1:]

            # making context and target for bwd lstm
            indices.reverse()
            self.ctx_b += list(indices[:i] for i in range(1, len(indices)))
            self.words_b += indices[1:]

            x = len(indices)
            self.max_len = x-1 if x-1>self.max_len else self.max_len

        # padding
        self.ctx_f = [ctx + [self.vocab[self.specials[1]]]*(self.max_len-len(ctx)) for ctx in self.ctx_f]
        self.ctx_b = [ctx + [self.vocab[self.specials[1]]]*(self.max_len-len(ctx)) for ctx in self.ctx_b]

        # conv to tensor
        self.words_f = tensor(self.words_f)
        self.words_b = tensor(self.words_b)
        self.ctx_f = tensor(self.ctx_f)
        self.ctx_b = tensor(self.ctx_b)

    def __getitem__(self, index):
        return self.ctx_f[index].to(device), self.words_f[index].to(device), self.ctx_b[index].to(device), self.words_b[index].to(device)

    def __len__(self):
        return len(2*self.words_f)

def get_embeddings(vocabulary):
    def get_unk(v):
        return torch.mean(v, dim=0)
    em_name = '6B'
    dim = 300
    glove = vocab.GloVe(name=em_name, dim=dim)
    unk_emb = get_unk(glove.vectors)
    embeds = list()
    for word in tqdm(vocabulary.get_itos(), desc='getting glove embeddings'):
        if word in glove.itos:
            embeds.append(glove[word])
        else:
            embeds.append(unk_emb)
    embeds = torch.stack(embeds)
    return embeds   
    
preprocessed_data = ProcessDataset("elmo_data/train.csv")
elmo_dataset_train = ElmoDataset(preprocessed_data.vocab, "elmo_data/train.csv")
elmo_dataset_val = ElmoDataset(preprocessed_data.vocab, 'elmo_data/train.csv')
embedding_matrix = get_embeddings(preprocessed_data.vocab)

mdl = Elmo(preprocessed_data.vocab, embedding_matrix).to(device)
train_loader = DataLoader(elmo_dataset_train, batch_size=32)
pad_idx = preprocessed_data.vocab['<PAD>']
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(mdl.parameters(), lr=1e-3)

num_epochs = 10

# training fn
for epoch in range(num_epochs):
    final_loss = 0
    for i, batch in enumerate(tqdm(train_loader, desc='training elmo')):
        i1,l1,i2,l2 = batch
        l1 = l1.unsqueeze(1)
        l2 = l2.unsqueeze(1)
        
        # transposing to make dims same
        i1 = torch.transpose(i1, 0,1)
        l1 = torch.transpose(l1, 0,1)
        i2 = torch.transpose(i2, 0,1)
        l2 = torch.transpose(l2, 0,1)
        
        total_loss = 0
        out1, out2 = mdl(i1,i2)
        length = len(l1)
        for x in range(length):
            loss_1 = loss_fn(out1[x], l1[x])
            loss_2 = loss_fn(out2[x], l2[x])
            total_loss += (loss_1 + loss_2) / 2
        optimiser.zero_grad()
        total_loss.backward()
        optimiser.step()

        final_loss += total_loss.item()
    avg_loss = final_loss / len(train_loader)
    final_loss = 0
    print(f"Average loss for epoch {epoch}: {avg_loss}")

torch.save(mdl.state_dict(), 'elmo_pretrained.pth')

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ProcessDataset():
    def __init__(self, filepath, min_freq=1):
        self.specials = ['<unk>, <pad>']
        df = pd.read_csv(filepath)
        all_words = list()
        for i in tqdm(range(len(df)), desc="building vocabulary"):
            if i < 7600:
                continue
            if i > 40000:
                break
            line = df['Description'][i]
            l = [[word.lower()] for word in word_tokenize(line)]
            all_words.extend(l)
            
        self.vocab = vocab.build_vocab_from_iterator(iterator=all_words, min_freq=min_freq, specials=self.specials)
        def_idx = self.vocab(self.specials[0])
        self.vocab.set_default_index(def_idx)
        
class ElmoDataset(Dataset):
    def __init__(self, vocabulary, filename, is_train=True):
        self.vocab = vocabulary
        self.ctx = list()
        self.words = list()
        self.max_len = -1
        self.df = pd.read_csv(filename)
        self.specials = ['<unk>', '<pad>']
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
                if i > 40000:
                    break
            else:
                if i > 7600:
                    continue
            
            # iterate over the dataset
            line = self.df['Description'][i]
            words = list()
            indices = list()
            for word in word_tokenize(line):
                words.append(word.lower())
            for word in words:
                indices.append(self.vocab[word])
                
            # making dataset from both front and back
            for i in range(1,3):
                if i % 2 == 0:
                    indices.reverse()
                self.ctx += list(indices[:i] for i in range(1, len(indices)))
                self.words += indices[1:]   
                
            x = len(indices)
            self.max_len = x-1 if x-1>self.max_len else self.max_len
            
        self.ctx = [ctx + self.specials[1]*(self.max_len-len(ctx)) for ctx in self.ctx]
        self.words = tensor(self.words)
        self.ctx = tensor(self.ctx)
        
    def __getitem__(self, index):
        return self.ctx[index].to(device), self.words.to(device) 
    
    def __len__(self):
        return len(self.words)
    
class ClassificationDataset(Dataset):
    def __init__(self,vocabulary, filename, is_train=True):
        self.vocab = vocabulary
        self.tokens = list()
        self.labels = list()
        self.max_len = -1
        self.df = pd.read_csv(filename)
        self.specials = ['<unk>', '<pad>']
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
                if i > 40000:
                    break
            else:
                if i > 7600:
                    continue
            
            # iterate over the dataset
            line = self.df['Description'][i]
            label = self.df['Class Index'][i]
            self.labels.append(label)
            words = list()
            indices = list()
            for word in word_tokenize(line):
                words.append(word.lower())
            for word in words:
                indices.append(self.vocab[word])
            self.tokens.append(indices) 
                
            x = len(indices)
            self.max_len = x if x>self.max_len else self.max_len
            
        self.tokens = [t + self.specials[1]*(self.max_len-len(t)) for t in self.tokens]
        self.labels = tensor(self.labels)
        self.tokens = tensor(self.tokens)
        
    def __getitem__(self, idx):
        return self.tokens[idx].to(device), self.labels[idx].to(device)
    def __len__(self):
        return len(self.labels)
    
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
    
        
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import Dataset, DataLoader
import os
import random
import regex as re
from collections import defaultdict
import sys
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from NNLM import NNLM

nltk.download('punkt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_text(filename):
    with open(filename, 'r') as f:
        para = str()
        sentences = list()
        for line in tqdm(f, desc="Splitting dataset"):
            if len(line.strip()) > 0:
                para += line.strip() + " "
            else:
                if para:
                    sentences += sent_tokenize(para)
                para = ""

        # Tokenize the last paragraph if it's not empty
        if para:
            sentences += sent_tokenize(para)
    return sentences

def make_split(sentences):
    random.shuffle(sentences)
    valid_len = 10000
    test_len = 10000
    train_len = len(sentences) - valid_len - test_len
    
    # write train data to a file
    with open('data/train.txt','w') as f:
        f.writelines([s + '\n' for s in sentences[:train_len]])
    # write valid data to a file
    with open('data/valid.txt','w') as f:
        f.writelines([s + '\n' for s in sentences[train_len:train_len+valid_len]])
    # write test data to a file
    with open('data/test.txt','w') as f:
        f.writelines([s + '\n' for s in sentences[train_len+valid_len:]])

def get_embeddings(filename):
    def find_unk(token='unk'):
        with open(filename, 'r') as f:
            for line in tqdm(f, desc='getting unk embedding'):
                if(line.strip().split()[0] == token):
                    e = line.strip().split()[1:]
                    return tensor([float(x) for x in e])
    unk_str = find_unk()
    embeddings = defaultdict(lambda: unk_str)
    with open(filename, 'r') as f:
        for line in tqdm(f, desc='getting embeddings'):
            line = line.strip().split()
            embeddings[line[0]] = tensor([float(x) for x in line[1:]])
    return embeddings

class Data(Dataset):
    def __init__(self, filepath, embeddings, context=5, vocab=None, freq_cutoff=1):
        self.filepath = filepath
        self.embeddings = embeddings
        self.frequency_cutoff = freq_cutoff
        self.context_size = context
        
        self.words = list()
        self.context_for_words = list()
        self.freq_dictionary = defaultdict(lambda:0)
        self.vocab = vocab if vocab is not None else list()
        self.word2idx = dict()
        self.unk_token = '<unk>'
        self.embeddings_list = list()
        
        with open(self.filepath, 'r') as f:
            for line in tqdm(f, desc='building vocab'):
                words = list()
                for word in word_tokenize(line):
                    word = word.lower()
                    words.append(word)
                    self.freq_dictionary[word] += 1
                if not vocab: 
                    self.vocab.extend(words)
            # for train set when no vocab provided
            if not vocab:
                self.vocab = list(set(self.vocab))
                for word in self.vocab:
                    if self.freq_dictionary[word] <= self.frequency_cutoff:
                        self.vocab.remove(word)
                self.vocab.append(self.unk_token)
            self.word2idx = {word:idx for idx, word in enumerate(self.vocab)}
            for word in self.vocab:
                self.embeddings_list.append(self.embeddings[word])
            self.embeddings_list.append(self.embeddings[self.unk_token])
            self.embeddings = torch.stack(self.embeddings_list)
            
        with open(self.filepath, 'r') as f:
            for line in tqdm(f, desc='building vocab'):
                words = list()
                for word in word_tokenize(line):
                    words.extend(word.lower())
                indices = list()
                for word in words:
                    if word in self.vocab:
                        indices.append(self.word2idx[word])
                    else:
                        indices.append(self.word2idx[self.unk_token])
                embeds = list()
                for i in indices:
                    embeds.append(self.embeddings[i])
                
                for i in range(len(embeds) - self.context_size):
                    self.words.append(indices[i+self.context_size])
                    self.context_for_words.append(torch.stack(embeds[i:i+self.context_size]))
                    
        self.context_for_words = torch.stack(self.context_for_words)
        self.words = torch.tensor(self.words)
    
    def __getitem__(self, index):
        return (self.context_for_words[index], self.words[index])
    
    def __len__(self):
        return len(self.words)       
        

# change file paths when running on colab
if __name__ == '__main__':
    data_file = 'data/Auguste_Maquet.txt'
    sentences=load_text(data_file)
    # print(len(sentences))
    make_split(sentences)
    embeddings_file = '/media/hitesh/DATA/IIIT-H/4th_year/Anlp/glove_embeddings/glove.6B.300d.txt'
    embeddings = get_embeddings(embeddings_file)
    train_file = 'data/train.txt'
    train_dataset = Data(filepath=train_file, embeddings=embeddings)
    
    dev_file = 'data/valid.txt'
    dev_dataset = Data(filepath=dev_file, embeddings=embeddings, vocab=train_dataset.vocab)
    
    test_file = 'data/test.txt'
    test_dataset = Data(filepath=test_file, embeddings=embeddings, vocab=train_dataset.vocab)
    test_data = DataLoader(test_dataset, batch_size=256, shuffle=True)
    
    learning_rates = [0.001,0.01,0.1]
    dimensions = [50,100,200,300,400,500]
    
    with open('ffn.txt', 'w') as f:
        for learning_rate in learning_rates:
            for dimension in dimensions:
                lm = NNLM(len(train_dataset.vocab), h1=dimension, h2=dimension).to(device)
                lm.train(train_dataset, dev_dataset, lr=learning_rate)  
                print("Learning rate is {} and hidden size is {}".format(learning_rate, dimension))
                prp = lm.get_perplexity(test_data)
                f.write("lr={}\ths={}\tprp={}\n".format(learning_rate, dimension, prp))
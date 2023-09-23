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
from nnlm import NNLM
import math

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
    valid_len = 8000
    test_len = 8000
    train_len = 30000
    
    # write train data to a file
    with open('data/train.txt','w') as f:
        f.writelines([s + '\n' for s in sentences[:train_len]])
    # write valid data to a file
    with open('data/valid.txt','w') as f:
        f.writelines([s + '\n' for s in sentences[train_len:train_len+valid_len]])
    # write test data to a file
    with open('data/test.txt','w') as f:
        f.writelines([s + '\n' for s in sentences[train_len+valid_len:train_len+valid_len+test_len]])

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
            # building vocab and frequency dict
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
                self.vocab = list(set(self.vocab)) # removing duplicates
                self.vocab.append(self.unk_token)
            # mapping word to indices
            self.word2idx = {word:idx for idx, word in enumerate(self.vocab)}
            # map words to indices and store in a list
            for word in self.vocab:
                self.embeddings_list.append(self.embeddings[word])
            self.embeddings = torch.stack(self.embeddings_list)
            
        with open(self.filepath, 'r') as f:
            for line in tqdm(f, desc='building vocab'):
                words = list()
                # convert list of words to list of indices using word2idx
                for word in word_tokenize(line):
                    words.append(word.lower())
                indices = list()
                for word in words:
                    if word in self.vocab:
                        indices.append(self.word2idx[word])
                    else:
                        indices.append(self.word2idx[self.unk_token])
                embeds = list()
                # get embeddings using indices
                for i in indices:
                    embeds.append(self.embeddings[i])
                # make input-output (5 gram and target word)
                for i in range(len(embeds) - self.context_size):
                    self.words.append(indices[i+self.context_size])
                    self.context_for_words.append(torch.stack(embeds[i:i+self.context_size]))
                    
        self.context_for_words = torch.stack(self.context_for_words)
        self.words = torch.tensor(self.words)
    
    def __getitem__(self, index):
        return (self.context_for_words[index], self.words[index])
    
    def __len__(self):
        return len(self.words) 
    
def get_perp_file(model, dataset, in_file, out_file):
    loss_fn = nn.CrossEntropyLoss()
    with open(in_file,'r') as f:
        with open(out_file, 'w') as g:
            for line in tqdm(f):
                words = list()
                for word in word_tokenize(line):
                    words.append(word.lower())
                indices = list()
                for word in words:
                    if word in dataset.vocab:
                        indices.append(dataset.word2idx[word])
                    else:
                        indices.append(dataset.word2idx[dataset.unk_token])
                embeds = list()
                for i in indices:
                    embeds.append(dataset.embeddings[i])
                    
                words = list()
                contexts = list()
                x = dataset.context_size
                for i in range(len(embeds) - x):
                    contexts.append(torch.stack(embeds[i:i+x]))
                    words.append(indices[i+x])
                
                words = torch.tensor(words)
                contexts = torch.stack(contexts)
                words = words.to(device)
                contexts = contexts.to(device)
                outs = model(contexts)
                outs = outs.view(-1, outs.shape[-1])
                words = words.view(-1)
                loss = loss_fn(outs, words)
                loss = loss.item()
                prp = math.exp(loss)
                s = line[:-1] + '\t' + str(prp) + '\n'
                g.write(s)
                            

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
    
    # lm = NNLM(len(train_dataset.vocab), h1=500, h2=500).to(device)
    # lm.train(train_dataset, dev_dataset, lr=0.1)
    # torch.save(lm, 'nnlm.pth')
    
    # lm = torch.load('nnlm.pth')
    # lm = lm.to(device)
    # print('generating prp files')
    # get_perp_file(lm, train_dataset,train_file,'2020115003-LM1-train-perplexity.txt')
    # get_perp_file(lm, test_dataset,test_file,'2020115003-LM1-test-perplexity.txt')
    
    
    learning_rates = [0.5,0.1,0.05]
    dimensions = [50,100,200,300,500, 750]
    
    with open('ffn.txt', 'w') as f:
        for learning_rate in learning_rates:
            for dimension in dimensions:
                lm = NNLM(len(train_dataset.vocab), h1=dimension, h2=dimension).to(device)
                lm.train(train_dataset, dev_dataset, lr=learning_rate)  
                print("Learning rate is {} and hidden size is {}".format(learning_rate, dimension))
                prp = lm.get_perplexity(test_data)
                f.write("lr={}-hs={}:\t{}\n".format(learning_rate, dimension, prp))
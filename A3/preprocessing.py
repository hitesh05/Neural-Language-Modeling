import torch
import torch.nn as nn
from torch import tensor
from torchtext import vocab
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Process_Dataset:
    def __init__(self, file_1, file_2, min_freq=5):
        self.file_en = file_1
        self.file_fr = file_2
        self.specials = ['<UNK>', '<PAD>']
        self.min_freq = min_freq
        self.process()
        
    def tok_en(self, sent):
        return [[word.lower()] for word in word_tokenize(sent, language='english')]
    
    def tok_fr(self, sent):
        return [[word.lower()] for word in word_tokenize(sent, language='french')]
    
    def process(self):
        # getting english vocab
        all_words = []
        with open(self.file_en, 'r') as f:
            for line in f.readlines():
                all_words.extend(self.tok_en(line))
        self.en_vocab = vocab.build_vocab_from_iterator(all_words, min_freq=self.min_freq, specials=self.specials)
        def_idx = self.en_vocab[self.specials[0]]
        self.en_vocab.set_default_index(def_idx)
        
        # getting french vocab
        all_words = []
        with open(self.file_fr, 'r') as f:
            for line in f.readlines():
                all_words.extend(self.tok_fr(line))   
        self.fr_vocab = vocab.build_vocab_from_iterator(all_words, min_freq=self.min_freq, specials=self.specials)
        def_idx = self.fr_vocab[self.specials[0]]
        self.fr_vocab.set_default_index(def_idx)
            
class TransformerData(Dataset):
    def __init__(self, en_vocab, fr_vocab, en_file, fr_file):
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab
        
        with open(en_file, 'r') as f:
            self.en_sents = f.readlines()
        with open(fr_file, 'r') as f:
            self.fr_sents = f.readlines()
        
        self.src_sents = []
        self.tgt_sents = []
        self.max_len = -1
        self.specials = ['<UNK>', '<PAD>']
        self.make_dataset()
        
    def make_dataset(self):
        for sent in tqdm(self.en_sents, desc='tokenising english sents'):
            words = [word.lower() for word in word_tokenize(sent, language='english')]
            indices = [self.en_vocab[word] for word in words]
            self.src_sents.append(indices)
            x = len(indices)
            self.max_len = x if x > self.max_len else self.max_len
                
        for sent in tqdm(self.fr_sents, desc='tokinsing fr sents'):
            words = [word.lower() for word in word_tokenize(sent, language='french')]
            indices = [self.fr_vocab[word] for word in words]
            self.tgt_sents.append(indices)
            x = len(indices)
            self.max_len = x if x > self.max_len else self.max_len
        
        # padding  
        self.src_sents = [sent + [self.en_vocab[self.specials[1]]]*(self.max_len - len(sent)) for sent in self.src_sents]
        self.tgt_sents = [sent + [self.fr_vocab[self.specials[1]]]*(self.max_len - len(sent)) for sent in self.tgt_sents]
        
    def __getitem__(self, index):
        return tensor(self.src_sents[index]).to(device), tensor(self.tgt_sents[index]).to(device)
    
    def __len__(self):
        return len(self.src_sents)
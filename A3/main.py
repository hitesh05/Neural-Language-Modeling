import torch
import torch.nn as nn
from torch import tensor
from torchtext import vocab
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from preprocessing import Process_Dataset
from preprocessing import TransformerData
from preprocessing import get_embeddings
from transformer import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

en_train_file = 'tranformers_data/train.en'
fr_train_file = 'transformers_data/train.fr'
process_dataset = Process_Dataset(en_train_file, fr_train_file)
train_data = TransformerData(process_dataset.en_vocab, process_dataset.fr_vocab, en_train_file, fr_train_file)
train_loader = DataLoader(train_data, batch_size=16)

en_embeddings = get_embeddings(process_dataset.en_vocab)
fr_embeddings = get_embeddings(process_dataset.fr_vocab)

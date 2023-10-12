import torch
import torch.nn as nn
from torch import tensor
from torchtext import vocab
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
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

en_vocab = process_dataset.en_vocab
fr_vocab = process_dataset.fr_vocab

train_data = TransformerData(en_vocab, fr_vocab, en_train_file, fr_train_file)
train_loader = DataLoader(train_data, batch_size=16)

en_embeddings = get_embeddings(process_dataset.en_vocab)
fr_embeddings = get_embeddings(process_dataset.fr_vocab)

d_model = 300
num_heads = 6
num_layers = 6
exp_factor = 2
max_len = train_data.max_len

LEARNING_RATE = 1e-3
NUM_EPOCHS = 5

model = Transformer(len(en_vocab), len(fr_vocab), d_model, num_heads, num_layers, exp_factor, max_len)
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
pad_idx = en_vocab['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

model.train()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for batch in tqdm(train_loader, desc='training transformer'):
        optimiser.zero_grad()
        src, tgt = batch
        outs = model(src, tgt[:, :-1])
        outs = outs.contiguous().view(-1, len(fr_vocab))
        tgt = tgt[:, 1:].contiguous().view(-1)
        loss = criterion(outs, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        
        epoch_loss += loss.item()
    train_loss = epoch_loss/len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}')
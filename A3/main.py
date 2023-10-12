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
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_loop(model, optimiser, criterion, num_epochs, train_loader):
    model.train()
    for epoch in range(num_epochs):
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
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='transformer model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for the model')
    parser.add_argument('--heads', type=int, default=6, help='number of attn heads')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--eps', type=int, default=10, help='number of epochs')
    parser.add_argument('--layers', type=int, default=6, help='number of encoder and decoder layers of the transformer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout for the model')
    args = parser.parse_args()
    print(f'lr={args.lr}_heads={args.heads}_bs={args.bs}_eps={args.eps}_layers={args.layers}_dropout={args.dropout}')
    
    print('building vocab')
    en_train_file = 'transformers_data/train.en'
    fr_train_file = 'transformers_data/train.fr'
    process_dataset = Process_Dataset(en_train_file, fr_train_file)

    en_vocab = process_dataset.en_vocab
    fr_vocab = process_dataset.fr_vocab

    train_data = TransformerData(en_vocab, fr_vocab, en_train_file, fr_train_file)

    en_embeddings = get_embeddings(process_dataset.en_vocab)
    fr_embeddings = get_embeddings(process_dataset.fr_vocab)

    d_model = 300
    exp_factor = 2
    num_heads = args.heads
    num_layers = args.layers
    batch_size = args.bs
    max_len = train_data.max_len
    lr = args.lr
    num_epochs = args.eps
    dropout = args.dropout

    train_loader = DataLoader(train_data, batch_size=batch_size)
    model = Transformer(len(en_vocab), len(fr_vocab), d_model, num_heads, num_layers, exp_factor, max_len, dropout).to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    pad_idx = en_vocab['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    print('training model')
    train_loop(model, optimiser, criterion, num_epochs, train_loader)
    
    print('saving model')
    torch.save(model.state_dict(), f'models/model_{lr}_{batch_size}_{dropout}.pth')
    
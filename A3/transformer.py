import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(max_len, 1, d_model) # stores pos encodings
        
        pe[:, 0, 0::2] = torch.sin(position * div_term) # even pos are filled with sin
        pe[:, 0, 1::2] = torch.cos(position * div_term) # odd pos are filled with cos
        self.register_buffer('pe', pe) # buffers are non-trainable params

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return x + self.pe[:x.size(0)]
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        '''
        args:
        d_model: model dimension/enbedding dims
        num_heads: number of attn heads
        '''
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model # 512
        self.num_heads = num_heads # 8
        self.d_k = d_model // num_heads # 512/8 = 64: each k,q,v will be 64 dim
        
        self.W_q = nn.Linear(self.d_model, self.d_model, bias=False) # 512x512: Single Query matrix for 8 attn heads
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=False) # key matrix
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=False) # value matrix
        self.W_o = nn.Linear(self.d_model, self.d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        K_transpose = torch.transpose(K, -2, -1) # calculate K transpose: [32,8,64,10]
        '''
        qkt = Q*(K^T)
        [32,8,10,64]x[32,8,64,10] = [32,8,10,10]
        '''
        qkt = torch.matmul(Q, K_transpose) # Key, Query products
        attn_scores = qkt / math.sqrt(self.d_k) # QK^T / 8, 8 is the sqrt of 64 (dim of single head)
        
        # fill those pos of attn_scores matrix where mask == 0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-1e20"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        '''
        output = attn_score * val
        [32,8,10,10]x[32,8,10,64] = [32,8,10,64]
        '''
        output = torch.matmul(attn_probs, V)
        return output # [32,8,10,64]
        
    def split_heads(self, x):
        batch_size, seq_length, _ = x.size() # [32,10,512]
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k) # [32,10,8,64]
        x = torch.transpose(x, 1, 2) # [32,8,10,64]
        return x
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size() # [32,8,10,64]
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model) # [32,10,512]
        return x
        
    def forward(self, Q, K, V, mask=None):
        x = self.W_q(Q)
        Q = self.split_heads(x) # [32,8,10,64]
        x = self.W_k(K)
        K = self.split_heads(x) # [32,8,10,64]
        x = self.W_v(V)
        V = self.split_heads(x) # [32,8,10,64]
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask) # attn_output = [32,8,10,64]
        attn_output = self.combine_heads(attn_output) # [32,10,512]
        output = self.W_o(attn_output) # [32,10,512]
        return output
    
class FFN(nn.Module):
    def __init__(self, d_model, exp_factor):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model*exp_factor)
        self.fc2 = nn.Linear(d_model*exp_factor, d_model)
        self.gelu = nn.GELU() # activation fn
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
    
class Encoder_Layer(nn.Module):
    def __init__(self, d_model, num_heads, exp_factor, dropout):
        super(Encoder_Layer, self).__init__()
        
        self.self_attn_layer = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, exp_factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attn_output = self.self_attn_layer(x,x,x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x  
    
class Decoder_Layer(nn.Module):
    def __init__(self, d_model, num_heads, exp_factor, dropout):
        super(Decoder_Layer, self).__init__()
        
        self.self_attn_layer = MultiHeadAttention(d_model, num_heads)
        self.cross_attn_layer = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, exp_factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, tgt_mask):
        attn_out = self.self_attn_layer(x,x,x,tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        attn_out = self.cross_attn_layer(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, exp_factor, max_seq_length, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([Encoder_Layer(d_model, num_heads, exp_factor, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([Decoder_Layer(d_model, num_heads, exp_factor, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
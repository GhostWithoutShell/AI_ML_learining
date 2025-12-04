import torch.nn as nn
import seaborn as sns
import numpy as np
import re
import torch
import math


class AttentionLayer(nn.Module):
    def __init__(self, size_kernel, num_heads, pad_index):
        super().__init__()
        self.size_kernel = size_kernel
        self.num_heads = num_heads
        self.pad_index = pad_index
        self.head_dim = size_kernel // num_heads

        self.key = nn.Linear(size_kernel, size_kernel)
        self.value = nn.Linear(size_kernel, size_kernel)
        self.query = nn.Linear(size_kernel, size_kernel)
        self.projection = nn.Linear(size_kernel, size_kernel)
    def forward(self, x, input_ids):
        batch_size, seq_len, _ = x.size()
        x_k = self.key(x)
        x_q = self.query(x)
        x_v = self.value(x)
        x_k = x_k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x_v = x_v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x_q = x_q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x_v_tr = x_v.transpose(1,2)
        x_q_tr = x_q.transpose(1,2)
        scores = torch.matmul(x_q_tr, x_k.transpose(-1,-2))
        padding_mask = (input_ids != self.pad_index).unsqueeze(1).unsqueeze(2)
        # mask
        scores = scores/math.sqrt(self.head_dim)
        tri_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(0).bool()
        final_mask = padding_mask & tri_mask
        attention_scores = scores.masked_fill(final_mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        result = torch.matmul(attention_scores, x_v_tr)
        result = result.transpose(1,2).contiguous()
        result = result.view(batch_size, seq_len, self.size_kernel)
        x = self.projection(result)
        #result = scores / torch.sqrt(x_v.transpose(-1,-2)
        return x

class FeedForward(nn.Module):
    def __init__(self, size_kernel, expansion=4):
        super().__init__()
        self.lin = nn.Linear(size_kernel*expansion, size_kernel)
        self.lin1 = nn.Linear(size_kernel, size_kernel*expansion)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(size_kernel)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = self.lin(x)
        x = self.norm(self.dropout(x))
        x = self.relu(x)
        x = self.lin1(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, size_kernel, num_heads, pad_index):
        super().__init__()
        self.attention = AttentionLayer(size_kernel, num_heads, pad_index)
        
        
        self.norm = nn.LayerNorm(size_kernel)
        self.dropout = nn.Dropout(0.2)
        self.ff = FeedForward(size_kernel=size_kernel)

    def forward(self, x, input_ids):
        
        x_att = self.attention(x, input_ids)
        x = self.norm(self.dropout(x_att)) + x
        x = self.ff(x)
        return x
    

    
class GPTLikeModel(nn.Module):
    def __init__(self, vocab_size, size_kernel, num_heads, num_layers, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, size_kernel, padding_idx=pad_index)
        self.pos_emb = nn.Embedding(512, size_kernel)
        self.norm = nn.LayerNorm(size_kernel)
        self.attention_blocks = AttentionBlock(size_kernel, num_heads, pad_index)
        self.attention_blocks1 = AttentionBlock(size_kernel, num_heads, pad_index)
        self.attention_blocks2 = AttentionBlock(size_kernel, num_heads, pad_index)
        
        self.drop = nn.Dropout(0.2)
        self.fc_out = nn.Linear(size_kernel, vocab_size)
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, seq_len).to(input_ids.device)
        self.embedding.weight = self.fc_out.weight
        x = self.embedding(input_ids) + self.pos_emb(positions)
        x = self.norm(x)
        x = self.attention_blocks(x, input_ids)
        x = self.attention_blocks1(x, input_ids)
        x = self.attention_blocks2(x, input_ids)
        logits = self.fc_out(self.drop(x))
        return logits
    


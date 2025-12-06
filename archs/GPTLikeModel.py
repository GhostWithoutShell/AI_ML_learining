import torch.nn as nn
import seaborn as sns
import numpy as np
import re
import torch
import math
from model_train_tools import DataBuilderImdb

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
        # mask
        padding_mask = (input_ids != self.pad_index).unsqueeze(1).unsqueeze(2)
        tri_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).unsqueeze(0).unsqueeze(0).bool()
        final_mask = padding_mask & tri_mask

        scores = scores/math.sqrt(self.head_dim)
        
        attention_scores = scores.masked_fill(final_mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        result = torch.matmul(attention_scores, x_v_tr)
        result = result.transpose(1,2).contiguous()
        result = result.view(batch_size, seq_len, self.size_kernel)
        x = self.projection(result)
        return x

class FeedForward(nn.Module):
    def __init__(self, size_kernel, expansion=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size_kernel, size_kernel*expansion),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(size_kernel*expansion, size_kernel)
        )
    def forward(self, x):
        x = self.layers(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, size_kernel, num_heads, pad_index):
        super().__init__()
        self.attention = AttentionLayer(size_kernel, num_heads, pad_index)
        
        
        self.norm = nn.LayerNorm(size_kernel)
        self.dropout = nn.Dropout(0.2)
        self.ff = FeedForward(size_kernel=size_kernel)

    def forward(self, x, input_ids):
        x_norm = self.norm(x)
        x_att = self.attention(x_norm, input_ids)
        x = self.dropout(x_att) + x
        x_out = self.ff(x)
        x = self.dropout(x_out) + x
        return x
    

    
class GPTLikeModel(nn.Module):
    def __init__(self, vocab_size, size_kernel, num_heads, num_layers, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, size_kernel, padding_idx=pad_index)
        self.pos_emb = nn.Embedding(512, size_kernel)
        self.norm = nn.LayerNorm(size_kernel)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(size_kernel, num_heads, pad_index) for _ in range(num_layers)
        ])        
        self.drop = nn.Dropout(0.2)
        self.fc_out = nn.Linear(size_kernel, vocab_size, bias=False)
        self.embedding.weight.data =self.embedding.weight
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, seq_len).to(input_ids.device)
        x = self.embedding(input_ids) + self.pos_emb(positions)
        x = self.norm(x)
        for block in self.attention_blocks:
            x = block(x, input_ids)
        logits = self.fc_out(self.drop(x))
        return logits
    

## train_input = input_ids[:, :-1]
## train_targets = input_ids[:, 1:]
dt = DataBuilderImdb()
data = dt.getDataFromCsv("D:\MyFiles\Datasets\IMDB\IMDB Dataset.csv")
params = {
    "textsColumn": "review",
    "labelsColumn": "sentiment"
}
data = dt.cleanTextFromTrash(data, params)
data = dt.applyLabelFix(data, params)
print(data.head())

num_epochs = 10
batch_size = 32
learning_rate = 0.001
vocab_size = 30000



model = GPTLikeModel(vocab_size=vocab_size, size_kernel=256, num_heads=8, num_layers=3, pad_index=0)



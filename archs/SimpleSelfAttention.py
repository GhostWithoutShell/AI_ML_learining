import pandas as pd
import torch
#import torchtext
#from torchtext.data.utils import get_tokenizer
#from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import math


class SimpleAttention(nn.Module):
    def __init__(self, size_kernel):
        super().__init__()
        self.size = size_kernel
        self.key = nn.Linear(size_kernel, int(size_kernel/2))
        self.value = nn.Linear(size_kernel, int(size_kernel/2))
        self.query = nn.Linear(size_kernel, int(size_kernel/2))
    def forward(self, x):
        x_k = self.key(x)
        x_v = self.value(x)
        x_q = self.query(x)


        print(f"Shape v :{x_v.shape}")
        print(f"Shape q :{x_q.shape}")
        print(f"Shape k :{x_k.shape}")

        transpose_k = torch.transpose(x_k, -2, -1)
        
        attention_score = torch.matmul(x_q, transpose_k)
        print(f"Transpose {transpose_k.shape}")
        scaled_scores = attention_score/math.sqrt(self.size)
        print(f"Scaled scores {scaled_scores}, shape {scaled_scores.shape}")
        att_weight = torch.softmax(scaled_scores, dim=1)
        print(f"Att weight {att_weight}, shape {att_weight.shape}")
        result_mat = torch.matmul(att_weight, x_v)
        return torch.mean(result_mat, dim=1)


class TransformerClass(nn.Module):
    def __init__(self, vocab_size, embeding_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embeding_dim)
        self.attention = SimpleAttention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.lin = nn.Linear(hidden_size//2, 1)
    def forward(self, x):
        x = self.emb(x)
        x = self.attention(x)
        x = self.norm(x)
        x = self.lin(x)
        return x
        
## tests
def test_attention():
    batch_size, seq_len, emb_dim = 2, 4, 8  # Маленькие размеры для дебага
    x = torch.randn(batch_size, seq_len, emb_dim)
    
    attention = SimpleAttention(emb_dim)  # Какой параметр должен быть?
    
    try:
        output = attention(x)
        print(f"Success! Output shape: {output.shape}")
    except Exception as e:
        print(f"Error: {e}")
        # Анализируй ошибку!
test_attention()
import pandas as pd
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt


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
        x_q = self.value(x)


        transpose_k = torch.transpose(self.x_k, -2, -1)
        
        attention_score = torch.mm(self.x_q, transpose_k)
        scaled_scores = attention_score/int(self.size**0.5)
        att_weight = torch.softmax(scaled_scores)
        result_mat = att_weight * x_v
        return torch.max(result_mat, dim=1)


class TransformerClass(nn.Module):
    def __init__(self, vocab_size, embeding_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embeding_dim)
        self.attention = SimpleAttention(vocab_size)
        feat_dim = vocab_size*vocab_size
        self.norm = nn.LayerNorm(feat_dim)
        self.lin = nn.Linear(feat_dim)
    def forward(self, x):
        x = self.emb(x)
        x = self.attention(x)
        x = self.norm(x)
        x = self.lin(x)
        return x
        


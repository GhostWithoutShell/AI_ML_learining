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
    def __init(self, size_kernel, vocab_size):
        super().__init__()
        self.size = size_kernel
        self.key = torch.Tensor(size_kernel, vocab_size)
        self.value = torch.Tensor(size_kernel, vocab_size)
        self.query = torch.Tensor(size_kernel, vocab_size)
    def forward(self, x):
        transpose_k = torch.transpose(self.key, 0, 1)
        attention_score = torch.mm(self.query, self.key)
        scaled_scores = attention_score/int(self.size**0.5)
        att_weight = torch.softmax(atte)
        return att_weight * self.value

class TransformerClass(nn.Module):
    def __init__(self, vocab_size, embeding_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embeding_dim)
        self.attention = SimpleAttention(vocab_size, hidden_size)
        feat_dim = hidden_size*hidden_size
        self.norm = nn.LayerNorm(feat_dim)
        self.lin = nn.Linear(feat_dim)
    def forward(self, x):
        x = self.emb(x)
        x = self.attention(x)
        x = self.norm(x)
        x = self.lin(x)
        return x
        


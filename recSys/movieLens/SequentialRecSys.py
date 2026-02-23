
import torch
import torch.nn as nn
import math
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch.nn.functional as F


class SeqRecModel(nn.Module):
    def __init__(self, num_items, hidden_dim_rnn, dense_hidden_dim):
        super().__init__()
        
        #self.gmf_user_emb = nn.Embedding(num_users, hidden_dim_rnn)
        self.item_emb = nn.Embedding(num_items, hidden_dim_rnn)

        
        self.rnn_ = nn.LSTM(input_size=hidden_dim_rnn, hidden_size=hidden_dim_rnn, batch_first=True, bidirectional=False)     

        input_dim = hidden_dim_rnn*2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dense_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dense_hidden_dim, dense_hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dense_hidden_dim//2, 1)
        )
        
    def forward(self, item_indices, target_indices):
        emb_items = self.item_emb(item_indices)
        emb_targets = self.item_emb(target_indices)

        x, _ = self.rnn_(emb_items)
        rnn_layer_last = x[:, -1, :]

        result = torch.cat([rnn_layer_last, emb_targets], dim=1)
        result_mlp = self.mlp(result)
        return result_mlp.squeeze()
    
class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x, y):
        diff = x-y
        ln_res = F.softplus(-diff)
        return ln_res.mean()
class SequentialDataset(Dataset):
    def __init__(self, user_history_dict, max_context_len, all_movies_ids, pad_id=0):
        self.all_movies_ids = all_movies_ids
        self.user_history = user_history_dict
        self.user_ids = list(user_history_dict.keys())
        self.max_context_len = max_context_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]      
        full_history = self.user_history[user_id]
        
        target = full_history[-1]
        source = full_history[:-1]
        real_size = len(source)
        if real_size < self.max_context_len:
            temp = []
            temp += [self.pad_id] * (self.max_context_len-real_size)
            temp.extend(source)
            source = temp
        elif real_size > self.max_context_len:
            source = source[-self.max_context_len:]
        negative_movie = np.random.choice(self.all_movies_ids)
        while negative_movie == target or negative_movie in source:
            negative_movie = np.random.choice(self.all_movies_ids)

        return {
            'input_ids' : torch.tensor(source, dtype = torch.long),
            'target_id' : torch.tensor(target, dtype = torch.long),
            'negative_id' : torch.tensor(negative_movie, dtype = torch.long)
        }

def MRR(model, test_dataset,targets, inputs, chunk_size): # mean reciprocal rank
    with torch.no_grad():
        model.eval()
        mrr_sum = 0.0
        num_k = 250
        #chunks = torch.split(all_targets, chunk_size)
        all_movie_ids = test_dataset.dataset.all_movies_ids

        for i in range(len(test_dataset)):
            history = inputs[i]
            samples = random.sample(all_movie_ids, num_k)
            samples[0] = targets[i].item()
            history_repeated = history.unsqueeze(0).repeat(len(samples), 1).to(device)
            score = model(history_repeated, torch.tensor(samples, dtype=torch.long).to(device))
            rank = (score > score[0]).sum().item() + 1
            repousal_rank = 1/rank
            mrr_sum += repousal_rank

        return mrr_sum / len(test_dataset)


encoder = LabelEncoder()
encoder_user = LabelEncoder()




dt = pd.read_csv('D://MyFiles//MLLEarning//AI_ML_learining//recSys//Datasets//movie_lens//rating.csv', usecols=['movieId', 'userId', 'rating', 'timestamp'])

df_genres = pd.read_csv('D://MyFiles//MLLEarning//AI_ML_learining//recSys//Datasets//movie_lens//movie.csv', usecols=['movieId', 'genres'])

dt = dt.merge(df_genres, on='movieId', how='left')

#len_ = 50000

#dt = dt[:len_]
genres = set()

genres = dt['genres'].dropna().str.split('|').explode().unique().tolist()
genres_to_idx = {genre : i for i, genre in enumerate(genres)}
max_len_genres = len(genres)

def prepare_list_genres(genre_string):
    vector = np.zeros(len(genres_to_idx), dtype=np.float32)
    if not isinstance(genre_string, str):
        return vector
    current_movies_genres = genre_string.split('|')

    for genre in current_movies_genres:
        if genre in genres_to_idx:
            idx = genres_to_idx[genre]
            vector[idx] = 1.0
    return vector


print(max_len_genres)
PAD_INDEX = 0
encoder.fit(dt['movieId'])
encoder_user.fit(dt['userId'])
len_ = 130000

dt['movieId_encoded'] = encoder.transform(dt['movieId'])
dt['movieId_encoded'] = dt['movieId_encoded'] + 1
dt['userId_encoded'] = encoder_user.transform(dt['userId'])
dt['genres_list'] = dt['genres'].apply(prepare_list_genres)
user_group = dt.sort_values(by='timestamp').groupby('userId_encoded')['movieId_encoded'].apply(list)
user_group = user_group[user_group.apply(len) > 1]
sequences = user_group.to_dict()

dataset = SequentialDataset(sequences, 4, dt['movieId_encoded'].unique().tolist(), pad_id=PAD_INDEX)
sequences_list = list(sequences.items())
proportions = [.75, .10, .15]
lengths = [int(p * len(dataset)) for p in proportions]
lengths[-1] = len(dataset) - sum(lengths[:-1])

num_epochs = 5
lr = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SeqRecModel(dt['movieId_encoded'].max() + 1, 64, 32)
model = model.to(device)
print(len(dataset))
optimizer = torch.optim.AdamW(model.parameters() ,lr = lr, weight_decay=1e-4)
generator_ = torch.Generator().manual_seed(42)
train_data, vl_dataset, test_data = random_split(dataset, lengths)
#train_data, val_data, test_data = random_split(dataset, lengths, generator=generator_)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
loss = BPRLoss()

best_loss = math.inf
for i in range(num_epochs):
    model.train()
    
    bpr = 0
    for batch in train_dataloader:
    
        history = batch['input_ids']
        neg_items = batch['negative_id']
        target_items = batch['target_id']

        history = history.to(device)
        neg_items = neg_items.to(device)
        target_items = target_items.to(device)
        output_pos = model(history, target_items)
        output_neg = model(history, neg_items)


    
        bpr = loss(output_pos, output_neg)
        optimizer.zero_grad()
        bpr.backward() 
        optimizer.step()

    if best_loss == math.inf:
        best_loss = bpr.item()
    if best_loss > bpr.item():
        best_loss = bpr.item()
        torch.save(model.state_dict(), 'best_seqmodel.pth')
        
        
        
    print(f"Epoch {i+1}, BPR losss : {bpr.item()}")

model.load_state_dict(torch.load('best_seqmodel.pth'))
targets = []
inputs = []


for i in range(len(test_data)):
    item = test_data[i]
    targets.append(item['target_id'])
    inputs.append(item['input_ids'])
print(len(test_data))
print(len(targets))
mrrResult = MRR(model=model,test_dataset=test_data,targets=targets, inputs=inputs, chunk_size=128)

print(f'MRR result {mrrResult}')

        




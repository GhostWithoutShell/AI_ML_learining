import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class RecSysBase(nn.Module):
    def __init__(self, user_emb_size, movie_emb_size, embeding_dim):
        super().__init__()
        self.embeding_user = nn.Embedding(user_emb_size, embeding_dim)
        self.embeding_film = nn.Embedding(movie_emb_size, embeding_dim)
        self.embeding_user_bias = nn.Embedding(user_emb_size, 1)
        self.embeding_item_bias = nn.Embedding(movie_emb_size, 1)
        
    def forward(self, user_id, film_id):
        user_vec = self.embeding_user(user_id)
        film_vec = self.embeding_film(film_id)
        user_bias = self.embeding_user_bias(user_id).squeeze()
        film_bias = self.embeding_item_bias(film_id).squeeze()
        
        temp_res = user_vec * film_vec
    
        return (temp_res.sum(1) + user_bias + film_bias)
class RecSysBaseMN(nn.Module):
    def __init__(self, user_emb_size, movie_emb_size, embeding_dim, reduce_num, len_genres, max_rating=5):
        super().__init__()
        self.max_rating = max_rating
        self.embeding_user = nn.Embedding(user_emb_size, embeding_dim)
        self.embeding_film = nn.Embedding(movie_emb_size, embeding_dim)
        
        self.lin1 = nn.Linear(embeding_dim*3, reduce_num)
        self.lin2 = nn.Linear(reduce_num, reduce_num//2)
        self.lin3 = nn.Linear(reduce_num//2, 1)
        self.lin_genres = nn.Linear(len_genres, embeding_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.2)
    def forward(self, user_id, film_id, genres=None):
        user_vec = self.embeding_user(user_id)
        
        film_vec = self.embeding_film(film_id)
        genres_x = self.lin_genres(genres)
        temp_res = torch.cat((user_vec, film_vec, genres_x), dim=1)
        
        x = self.lin1(temp_res)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.lin3(x).squeeze()
    
        return self.sigmoid(x) * self.max_rating
    
class MovieDataset(Dataset):
    def __init__(self, user_ids, movie_ids, ratings, genres):
        self.users = torch.tensor(user_ids, dtype=torch.long)
        self.movies = torch.tensor(movie_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.genres = torch.tensor(list(genres), dtype=torch.float32)
    def __len__(self):
        return len(self.ratings)
    def __getitem__(self, idx):
        return (self.users[idx], self.movies[idx], self.ratings[idx], self.genres[idx])



    
    

from sklearn.preprocessing import MultiLabelBinarizer
dt = pd.read_csv('Datasets//movie_lens//rating.csv', usecols=['movieId', 'userId', 'rating'])
df_genres = pd.read_csv('Datasets//movie_lens//movie.csv', usecols=['movieId', 'genres'])

dt = dt.merge(df_genres, on='movieId', how='left')

len_ = 20000

dt = dt[:len_]
genres = set()

genres = dt['genres'].dropna().str.split('|').explode().unique().tolist()
genres_to_idx = {genre : i for i, genre in enumerate(genres)}
max_len_genres = len(genres)
print(max_len_genres)


## Продакшен реди вариант
#mlb = MultiLabelBinarizer()
#temp_ser = dt['genres'].str.split('|')
#genre_vector = mlb.fit_transform(temp_ser)
#print(genre_vector)

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
    


dt['genres_list'] = dt['genres'].apply(prepare_list_genres)
print(dt.head())
generator = torch.Generator().manual_seed(42)


label_encoder_user = LabelEncoder()
label_encoder_movie = LabelEncoder()
print(dt.head())


label_encoder_user.fit(dt['userId'])
label_encoder_movie.fit(dt['movieId'])

print(label_encoder_user.classes_)

dt['userId_encoded'] = label_encoder_user.transform(dt['userId'])
dt['movieId_encoded'] = label_encoder_movie.transform(dt['movieId'])

dataset = MovieDataset(dt['userId_encoded'].values, dt['movieId_encoded'].values, dt['rating'].values, dt['genres_list'].values)

train_data, test_data = random_split(dataset, [int(len_*0.8), int(len_*0.2)], generator=generator)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RecSysBaseMN(user_emb_size=dt['userId_encoded'].nunique(),
                   movie_emb_size=dt['movieId_encoded'].nunique(),
                   embeding_dim=50, reduce_num=64, len_genres=max_len_genres).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
best_loss = float('inf')
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    loss = 0
    loss_val = 0
    losses = []
    for i, batch in enumerate(train_dataloader):
        users = batch[0]
        movies = batch[1]
        target = batch[2]
        genres = batch[3]
        optimizer.zero_grad()
        users = users.to(device)
        movies = movies.to(device)
        target = target.to(device)
        genres = genres.to(device)
        output = model(user_id=users, film_id=movies, genres=genres)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    all_preds = []
    loss_test = 0
    with torch.no_grad():
        for batch in test_dataloader:
            user, movie, target, genres = batch
            user = user.to(device)
            movie = movie.to(device)
            target = target.to(device)
            genres = genres.to(device)
            output = model(user_id = user, film_id = movie, genres=genres)
            loss = criterion(output, target)
            all_preds.append(loss.item())
            loss_test += loss.item()
            
        rms_loss_val = math.sqrt(loss_test/len(test_dataloader))
        if epoch == 0:
            best_loss = rms_loss_val
        if best_loss > rms_loss_val:
            print(f"New best model found at epoch {epoch} with loss {rms_loss_val}")
            torch.save(model.state_dict(), 'best_model.pth')
            best_loss = rms_loss_val
    
    print(f"Loss epoch : {math.sqrt(sum(losses)/len(losses))}, validation = {math.sqrt(loss_test/len(test_dataloader))}")
model = RecSysBaseMN(user_emb_size=dt['userId_encoded'].nunique(),
                   movie_emb_size=dt['movieId_encoded'].nunique(),
                   embeding_dim=50, reduce_num=64, len_genres=max_len_genres).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)

def recommended_for_user(model, dt, device, user_id_encoded, label_encoder_movie, movie_genre_dict, k=5):
    
    
    movies_all = list(movie_genre_dict.keys())
    
    
    movies_watched = dt[dt['userId'] == user_id_encoded]['movieId_encoded'].unique()
    
    
    candidates = list(set(movies_all) - set(movies_watched))
    
    if not candidates:
        print("User has seen all movies!")
        return

    
    
    candidate_genres_list = [movie_genre_dict[movie_id] for movie_id in candidates]
    
    
    movies_tensor = torch.tensor(candidates, dtype=torch.long).to(device)
    genres_tensor = torch.tensor(candidate_genres_list, dtype=torch.float32).to(device)
    
    
    user_tensor = torch.tensor([user_id_encoded] * len(candidates), dtype=torch.long).to(device)
    
    model.eval()
    with torch.no_grad():
        
        predictions = model(user_id=user_tensor, film_id=movies_tensor, genres=genres_tensor)
        
        
        values, top_indices = torch.topk(predictions, k)
        
        
        top_movie_ids_encoded = [candidates[i] for i in top_indices.cpu().numpy()]
        
        real_movie_ids = label_encoder_movie.inverse_transform(top_movie_ids_encoded)
        print(f"Top {k} recommendations for user {user_id_encoded}: {real_movie_ids}")
movie_genre_dict = dt.drop_duplicates('movieId_encoded').set_index('movieId_encoded')['genres_list'].to_dict()
recommended_for_user(model, dt, device, 42, label_encoder_movie, movie_genre_dict, k=5)




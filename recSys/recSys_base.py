import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
    def __init__(self, user_emb_size, movie_emb_size, embeding_dim, reduce_num, max_rating=5):
        super().__init__()
        self.max_rating = max_rating
        self.embeding_user = nn.Embedding(user_emb_size, embeding_dim)
        self.embeding_film = nn.Embedding(movie_emb_size, embeding_dim)
        
        self.lin1 = nn.Linear(embeding_dim+embeding_dim, reduce_num)
        self.lin2 = nn.Linear(reduce_num, reduce_num//2)
        self.lin3 = nn.Linear(reduce_num//2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.2)
    def forward(self, user_id, film_id):
        user_vec = self.embeding_user(user_id)
        film_vec = self.embeding_film(film_id)
        temp_res = torch.cat((user_vec, film_vec), dim=1)
        x = self.lin1(temp_res)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.lin3(x).squeeze()
    
        return self.sigmoid(x) * self.max_rating
    
class MovieDataset(Dataset):
    def __init__(self, user_ids, movie_ids, ratings):
        self.users = torch.tensor(user_ids, dtype=torch.long)
        self.movies = torch.tensor(movie_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
    def __len__(self):
        return len(self.ratings)
    def __getitem__(self, idx):
        return (self.users[idx], self.movies[idx], self.ratings[idx])
    
def recommended_for_user(model, dt, device, user_id_encoded, label_encoder_movie, k = 5):
    
    movies_all = dt['movieId_encoded'].unique()
    
    movies_watched = dt[dt['userId'] == user_id_encoded]['movieId_encoded']
    diff = list(set(movies_all) - set(movies_watched))
    
    
    values =torch.tensor(diff, dtype=torch.long)
    model.eval()
    user_tensor = torch.tensor([user_id_encoded]*len(values), dtype=torch.long).to(device)
    values = values.to(device)
    with torch.no_grad():
        predictions = model(user_id=user_tensor, film_id=values)
        values, top5_idx = torch.topk(predictions, k)
        recomended_films = [diff[i] for i in top5_idx.cpu().numpy()]
        real_movie_ids = label_encoder_movie.inverse_transform(recomended_films)
        print(f"Top {k} recommendations for user {user_id_encoded}:{real_movie_ids}")



dt = pd.read_csv('Datasets//movie_lens//rating.csv', usecols=['movieId', 'userId', 'rating'])
len_ = 200000
dt = dt[:len_]
generator = torch.Generator().manual_seed(42)


label_encoder_user = LabelEncoder()
label_encoder_movie = LabelEncoder()
print(dt.head())


label_encoder_user.fit(dt['userId'])
label_encoder_movie.fit(dt['movieId'])

print(label_encoder_user.classes_)

dt['userId_encoded'] = label_encoder_user.transform(dt['userId'])
dt['movieId_encoded'] = label_encoder_movie.transform(dt['movieId'])

dataset = MovieDataset(dt['userId_encoded'].values, dt['movieId_encoded'].values, dt['rating'].values)

train_data, test_data = random_split(dataset, [int(len_*0.8), int(len_*0.2)], generator=generator)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RecSysBaseMN(user_emb_size=dt['userId_encoded'].nunique(),
                   movie_emb_size=dt['movieId_encoded'].nunique(),
                   embeding_dim=50, reduce_num=64).to(device)
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
        optimizer.zero_grad()
        users = users.to(device)
        movies = movies.to(device)
        target = target.to(device)
        output = model(user_id=users, film_id=movies)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    all_preds = []
    loss_test = 0
    with torch.no_grad():
        for batch in test_dataloader:
            user, movie, target = batch
            user = user.to(device)
            movie = movie.to(device)
            target = target.to(device)
            output = model(user_id = user, film_id = movie)
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
                   embeding_dim=50, reduce_num=64).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
recommended_for_user(model, dt, device, 42, label_encoder_movie, k=5)




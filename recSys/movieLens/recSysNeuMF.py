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



class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, gmf_emb_dim, mlp_emb_dim, mlp_hidden_dims, len_genres, genres_emb):
        super().__init__()
        # gmf part
        self.gmf_user_emb = nn.Embedding(num_users, gmf_emb_dim)
        self.gmf_item_emb = nn.Embedding(num_items, gmf_emb_dim)

        # mlp part
        self.mlp_user_emb = nn.Embedding(num_users, mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(num_items, mlp_emb_dim)
        
        self.genres_lin = nn.Linear(len_genres, genres_emb)
        layers = []
        input_dim = mlp_emb_dim * 2 + genres_emb
        # создаем список слоев для обработки ML башенкой
        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
            
        self.mlp_layers = nn.Sequential(*layers)
        
        self.final_layer = nn.Linear(gmf_emb_dim + mlp_hidden_dims[-1], 1)
        
    def forward(self, user_indices, item_indices, genres_vec):
        x_user_gmf = self.gmf_user_emb(user_indices)
        y_item_gmf = self.gmf_item_emb(item_indices)
        x_gmf = x_user_gmf * y_item_gmf
        x_user_mlp = self.mlp_user_emb(user_indices)
        x_item_mlp = self.mlp_item_emb(item_indices)
        x_genres = self.genres_lin(genres_vec)

        x_mlp_layer = torch.cat([x_user_mlp, x_item_mlp, x_genres], dim=1)
        x_mlp_layer = self.mlp_layers(x_mlp_layer)

        vector = torch.cat([x_gmf, x_mlp_layer], dim=1)
        return self.final_layer(vector).squeeze()
    
        
    
class MovieDataset(Dataset):
    def __init__(self, user_ids, movie_ids, ratings, genres):
        self.users = torch.tensor(user_ids, dtype=torch.long)
        self.movies = torch.tensor(movie_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        self.genres = torch.tensor(np.stack(genres), dtype=torch.float32)
    def __len__(self):
        return len(self.ratings)
    def __getitem__(self, idx):
        return (self.users[idx], self.movies[idx], self.ratings[idx], self.genres[idx])

    def getAllMovies(self):
        return list(self.movies)

    
    

from sklearn.preprocessing import MultiLabelBinarizer
dt = pd.read_csv('D://MyFiles//MLLEarning//AI_ML_learining//recSys//Datasets//movie_lens//rating.csv', usecols=['movieId', 'userId', 'rating'])
df_genres = pd.read_csv('D://MyFiles//MLLEarning//AI_ML_learining//recSys//Datasets//movie_lens//movie.csv', usecols=['movieId', 'genres'])

dt = dt.merge(df_genres, on='movieId', how='left')



dt = dt[:20000]
genres = set()

genres = dt['genres'].dropna().str.split('|').explode().unique().tolist()
genres_to_idx = {genre : i for i, genre in enumerate(genres)}
max_len_genres = len(genres)



## Продакшен реди вариант
#mlb = MultiLabelBinarizer()
#temp_ser = dt['genres'].str.split('|')
#genre_vector = mlb.fit_transform(temp_ser)
#print(genre_vector)
def evaluate_one_case(model, user_id, true_item_id, all_movie_ids_set, user_interacted_items, movie_genre_dict, device, k=10):
    """
    user_id: int - ID пользователя
    true_item_id: int - ID фильма, который мы проверяем (Ground Truth)
    all_movie_ids_set: set - множество ВСЕХ возможных ID фильмов
    user_interacted_items: set - множество фильмов, которые этот юзер УЖЕ видел (история)
    """
    
    # 1. Негативный сэмплинг
    # Ищем фильмы, которые есть в базе, но юзер их не видел
    # ВАЖНО: исключаем true_item_id, чтобы случайно не добавить его как негатив
    possible_candidates = all_movie_ids_set - user_interacted_items - {true_item_id}
    
    # Берем 99 случайных
    negative_samples = random.sample(list(possible_candidates), 99)
    
    # 2. Формируем список: 99 плохих + 1 хороший (в конце)
    candidates = negative_samples + [true_item_id]
    target_idx = 99 # Индекс нашего правильного фильма (он последний)
    
    # 3. Подготовка данных
    candidate_genres_list = [movie_genre_dict[movie_id] for movie_id in candidates]
    
    user_tensor = torch.tensor([user_id] * 100, dtype=torch.long).to(device)
    movies_tensor = torch.tensor(candidates, dtype=torch.long).to(device)
    genres_tensor = torch.tensor(candidate_genres_list, dtype=torch.float32).to(device)
    
    # 4. Предикт
    model.eval()
    ndcg = 0
    result = []
    with torch.no_grad():
        output = model(user_indices=user_tensor, item_indices=movies_tensor, genres_vec=genres_tensor)
        if random.random() < 0.01: # 1% шанс срабатывания
            print(f"\n--- DEBUG CASE ---")
            print(f"Target Index: {target_idx}")
            print(f"Model Outputs (First 5): {output[:5].cpu().numpy()}")
            print(f"Model Output (Target): {output[target_idx].item()}")

            values, top_indices = torch.topk(output, k)
            print(f"Top K Indices: {top_indices.cpu().tolist()}")

            is_hit = target_idx in top_indices.cpu().tolist()
            print(f"Is Hit: {is_hit}")
            print("------------------\n")
        values, top_indices = torch.topk(output, k)
        # if target_idx in top_indices.cpu().numpy().tolist() else -1            
        if target_idx in top_indices.cpu().tolist():
            indx = top_indices.cpu().numpy().tolist().index(target_idx)
            indx = indx + 2
            indx = math.log2(indx)
            ndcg = 1 / math.log2(indx + 1)
            return (1, ndcg)
    
    return (0, ndcg)
def evaluate_global(model, test_dataset, all_movie_ids, user_interacted_dict, movie_genre_dict, device, k=10):
    hits = 0
    count = 0
    total_ndcg = 0
    all_movie_ids_set = set(all_movie_ids) # Превращаем в set один раз для скорости
    
    
    for user, movie, rating, _ in test_dataset:
        u_id = user.item()
        m_id = movie.item()
        
        # Пропускаем, если рейтинг в тесте низкий? (Опционально. Обычно HitRate считают на всем тесте или только на лайках)
        # Если хочешь считать HitRate только для "любимых" фильмов:
        if rating.item() < 4.0: 
             continue
             
        # Достаем историю ЭТОГО юзера
        interacted_items = set(user_interacted_dict.get(u_id, []))
        
        hit, ndcg = evaluate_one_case(
            model, 
            u_id, 
            m_id, 
            all_movie_ids_set, 
            interacted_items, 
            movie_genre_dict, 
            device, 
            k
        )
        hits += hit
        count += 1
        total_ndcg += ndcg
        if count % 100 == 0:
            print(f"Evaluated {count} cases...")
            
    return hits / count if count > 0 else 0, total_ndcg / count if count > 0 else 0
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
dt = dt[dt['rating'] >= 3.0].copy().reset_index(drop=True)
dt = dt[:16000]
generator = torch.Generator().manual_seed(42)


label_encoder_user = LabelEncoder()
label_encoder_movie = LabelEncoder()



label_encoder_user.fit(dt['userId'])
label_encoder_movie.fit(dt['movieId'])


len_ = len(dt)
print(int(len_*0.8), int(len_*0.2))
dt['userId_encoded'] = label_encoder_user.transform(dt['userId'])
dt['movieId_encoded'] = label_encoder_movie.transform(dt['movieId'])

dataset = MovieDataset(dt['userId_encoded'].values, dt['movieId_encoded'].values, dt['rating'].values, dt['genres_list'].values)

train_data, test_data = random_split(dataset, [int(len_*0.8), int(len_*0.2)], generator=generator)
movie_genre_dict = dt.drop_duplicates('movieId_encoded').set_index('movieId_encoded')['genres_list'].to_dict()
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuMF(num_users=dt['userId_encoded'].nunique(),
    num_items=dt['movieId_encoded'].nunique(),
    gmf_emb_dim = 32,
    mlp_emb_dim = 32,
    mlp_hidden_dims = [64,32,16],
    len_genres=max_len_genres,
    genres_emb=16).to(device)
bceCross = nn.BCEWithLogitsLoss()
all_movie_ids = dt['movieId_encoded'].unique()
user_interacted_dict = dt.groupby('userId_encoded')['movieId_encoded'].apply(set).to_dict()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
best_loss = float('inf')
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    loss = 0
    loss_val = 0
    losses = []
    for i, batch in enumerate(train_dataloader):
        user, movie, target, genres = batch
        batch_size = user.size(0)
        movie = movie.to(device)
        user = user.to(device)
        genres = genres.to(device)
        target = target.to(device)
        movie_neg = torch.randint(0, dt['movieId_encoded'].nunique(), (batch_size,)).to(device)
        candidate_genres_list = [movie_genre_dict[movie_id] for movie_id in movie_neg.cpu().numpy()]
        genres_neg = torch.tensor(candidate_genres_list, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        
        users_all = torch.cat([user, user]) # Юзер тот же
        movies_all = torch.cat([movie, movie_neg])
        genres_all = torch.cat([genres, genres_neg])

        users = users_all.to(device)
        movies = movies_all.to(device)
        genres = genres_all.to(device)
        target_pos = torch.ones(batch_size, dtype=torch.float32).to(device)
        target_neg = torch.zeros(batch_size, dtype=torch.float32).to(device)
        target = torch.cat([target_pos, target_neg])
        output = model(user_indices=users_all, item_indices=movies_all, genres_vec=genres_all)
        
        loss = bceCross(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    all_preds = []
    loss_test = 0
    with torch.no_grad():
        result = evaluate_global(model, test_dataset=test_data, all_movie_ids=all_movie_ids,user_interacted_dict=user_interacted_dict, movie_genre_dict=movie_genre_dict,device=device, k=10)
        if epoch == 0:
            best_loss = result
        if result > best_loss:
            print(f"New best model found at epoch {epoch} with loss {result}")
            torch.save(model.state_dict(), 'best_model.pth')
            best_loss = result
    
    print(f"Loss epoch : {sum(losses)/len(losses)}, validation = {result}")

model = NeuMF(num_users=dt['userId_encoded'].nunique(),
    num_items=dt['movieId_encoded'].nunique(),
    gmf_emb_dim = 32,
    mlp_emb_dim = 32,
    mlp_hidden_dims = [64,32,16],
    len_genres=max_len_genres,
    genres_emb=16).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)

def recommended_for_user(model, dt, device, user_id_encoded, label_encoder_movie, movie_genre_dict, k=5):
    movies_all = list(movie_genre_dict.keys())
    movies_watched = dt[dt['userId_encoded'] == user_id_encoded]['movieId_encoded'].unique()
    
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
        
        predictions = model(user_indices=user_tensor, item_indices=movies_tensor, genres_vec=genres_tensor)
        
        
        values, top_indices = torch.topk(predictions, k)
        
        
        top_movie_ids_encoded = [candidates[i] for i in top_indices.cpu().numpy()]
        
        real_movie_ids = label_encoder_movie.inverse_transform(top_movie_ids_encoded)
        print(f"Top {k} recommendations for user {user_id_encoded}: {real_movie_ids}")





    



#result = evaluate_global(model, test_dataset=test_data, all_movie_ids=all_movie_ids,user_interacted_dict=user_interacted_dict, movie_genre_dict=movie_genre_dict,device=device, k=10)
#print("Result ", result)



#recommended_for_user(model, dt, device, 42, label_encoder_movie, movie_genre_dict, k=5)



